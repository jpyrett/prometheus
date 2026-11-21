-- Prometheus account book (design choices entry 72).
-- Apply with scripts/setup_postgres.sh, which passes -v app_password=...
-- The password is not stored in this file. Re-running creates what is missing,
-- replaces the functions, and sets the application password again. It does not drop tables.

\if :{?app_password}
\else
\echo 'pass -v app_password=... (the value is in secrets/postgres_local.txt)'
\quit 1
\endif

BEGIN;

CREATE SCHEMA IF NOT EXISTS trading;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'prometheus_owner') THEN
    CREATE ROLE prometheus_owner NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION BYPASSRLS;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'prometheus_app') THEN
    CREATE ROLE prometheus_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS;
  END IF;
END
$$;

ALTER ROLE prometheus_owner WITH NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION BYPASSRLS;
ALTER ROLE prometheus_app WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS PASSWORD :'app_password';
ALTER ROLE prometheus_app SET search_path = trading;
ALTER SCHEMA trading OWNER TO prometheus_owner;

REVOKE ALL ON DATABASE prometheus FROM PUBLIC;
GRANT CONNECT ON DATABASE prometheus TO prometheus_app;
REVOKE ALL ON SCHEMA public FROM PUBLIC;

-- The owner is not a superuser. BYPASSRLS lets its functions maintain rows.
-- The application role is subject to the policies below. A query that has not
-- set trading.account_id sees nothing. That stops a forgotten filter in our code.
-- It is not a login: anyone who has the application password can set the id.

SET ROLE prometheus_owner;

CREATE OR REPLACE FUNCTION trading.current_account_id() RETURNS bigint
LANGUAGE sql
STABLE
SET search_path = trading, pg_temp
AS $$
  SELECT CASE
    WHEN current_setting('trading.account_id', true) ~ '^[0-9]+$'
      THEN current_setting('trading.account_id', true)::bigint
    ELSE NULL
  END
$$;

CREATE TABLE IF NOT EXISTS trading.accounts (
  account_id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  seed_key                  text UNIQUE,
  display_name              text NOT NULL CHECK (char_length(btrim(display_name)) BETWEEN 1 AND 200),
  preferred_name            text CHECK (preferred_name IS NULL OR char_length(preferred_name) BETWEEN 1 AND 80),
  email                     text CHECK (email IS NULL OR email ~ '^[^@[:space:]]+@[^@[:space:]]+\.[^@[:space:]]+$'),
  phone                     text CHECK (phone IS NULL OR phone ~ '^[0-9+(). -]{7,40}$'),
  timezone                  text NOT NULL DEFAULT 'America/New_York',
  currency                  text NOT NULL DEFAULT 'USD' CHECK (currency = 'USD'),
  status                    text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'closed')),
  risk_tolerance            text NOT NULL CHECK (risk_tolerance IN ('conservative', 'moderate', 'aggressive')),
  age                       integer CHECK (age IS NULL OR age BETWEEN 18 AND 120),
  investment_horizon_years  numeric(6, 2) CHECK (investment_horizon_years IS NULL OR investment_horizon_years > 0),
  trade_mode                text NOT NULL DEFAULT 'ask' CHECK (trade_mode IN ('ask', 'auto')),
  min_price                 numeric(14, 4) NOT NULL DEFAULT 5 CHECK (min_price >= 0),
  excluded_symbols          text[] NOT NULL DEFAULT '{}',
  excluded_sectors          text[] NOT NULL DEFAULT '{}',
  note                      text CHECK (note IS NULL OR char_length(note) <= 2000),
  cash_balance              numeric(14, 2) NOT NULL DEFAULT 0 CHECK (cash_balance >= 0),
  created_at                timestamptz NOT NULL DEFAULT now(),
  updated_at                timestamptz NOT NULL DEFAULT now()
);

-- Design choices entry 91 (Jacob, 2026-11-15): no profile limit applies to any order, so the four size limits are not stored any more. An account book made before that has the columns: drop them (idempotent).
ALTER TABLE trading.accounts
  DROP COLUMN IF EXISTS max_position_pct,
  DROP COLUMN IF EXISTS max_loss_per_trade_pct,
  DROP COLUMN IF EXISTS max_new_exposure_pct,
  DROP COLUMN IF EXISTS max_order_dollars;

-- Design choices entry 104 (Jacob, 2026-11-15): there is no second, automatic-trading mark: Auto mode trades what the Critic approves. An account book made before that has the column: drop it (idempotent).
ALTER TABLE trading.accounts
  DROP COLUMN IF EXISTS auto_min_score;

CREATE TABLE IF NOT EXISTS trading.cash_ledger (
  entry_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id    bigint NOT NULL REFERENCES trading.accounts (account_id) ON DELETE CASCADE,
  kind          text NOT NULL CHECK (kind IN ('deposit', 'withdrawal', 'buy', 'sell')),
  amount        numeric(14, 2) NOT NULL CHECK (amount <> 0),
  balance_after numeric(14, 2) NOT NULL CHECK (balance_after >= 0),
  memo          text,
  occurred_on   date NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  CHECK (
    (kind = 'deposit' AND amount > 0)
    OR (kind = 'withdrawal' AND amount < 0)
    OR (kind = 'buy' AND amount < 0)
    OR (kind = 'sell' AND amount > 0)
  )
);

CREATE INDEX IF NOT EXISTS cash_ledger_account_day ON trading.cash_ledger (account_id, occurred_on);

-- Cash moves only by inserting a ledger row. The row says the balance that follows,
-- and this trigger checks that arithmetic and writes accounts.cash_balance.
-- The application role cannot update cash_balance itself.
CREATE OR REPLACE FUNCTION trading.apply_ledger_entry() RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = trading, pg_temp
AS $$
DECLARE
  current_cash numeric(14, 2);
BEGIN
  SELECT cash_balance INTO current_cash FROM accounts WHERE account_id = NEW.account_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'no account %', NEW.account_id;
  END IF;
  IF NEW.balance_after IS DISTINCT FROM current_cash + NEW.amount THEN
    RAISE EXCEPTION 'ledger balance % does not follow cash % by %', NEW.balance_after, current_cash, NEW.amount;
  END IF;
  UPDATE accounts SET cash_balance = NEW.balance_after WHERE account_id = NEW.account_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'cash update was not applied for account %', NEW.account_id;
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS cash_ledger_apply ON trading.cash_ledger;
CREATE TRIGGER cash_ledger_apply
  AFTER INSERT ON trading.cash_ledger
  FOR EACH ROW
  EXECUTE FUNCTION trading.apply_ledger_entry();

CREATE OR REPLACE FUNCTION trading.cash_matches_ledger() RETURNS trigger
LANGUAGE plpgsql
SET search_path = trading, pg_temp
AS $$
DECLARE
  latest numeric(14, 2);
BEGIN
  IF NEW.cash_balance IS NOT DISTINCT FROM OLD.cash_balance THEN
    RETURN NEW;
  END IF;
  SELECT balance_after INTO latest
  FROM cash_ledger
  WHERE account_id = NEW.account_id
  ORDER BY entry_id DESC
  LIMIT 1;
  IF latest IS DISTINCT FROM NEW.cash_balance THEN
    RAISE EXCEPTION 'cash_balance (%) does not match the latest ledger row (%)', NEW.cash_balance, latest;
  END IF;
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION trading.touch_account() RETURNS trigger
LANGUAGE plpgsql
SET search_path = trading, pg_temp
AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS accounts_touch ON trading.accounts;
CREATE TRIGGER accounts_touch
  BEFORE UPDATE ON trading.accounts
  FOR EACH ROW
  EXECUTE FUNCTION trading.touch_account();

DROP TRIGGER IF EXISTS accounts_cash ON trading.accounts;
CREATE TRIGGER accounts_cash
  BEFORE UPDATE ON trading.accounts
  FOR EACH ROW
  EXECUTE FUNCTION trading.cash_matches_ledger();

CREATE TABLE IF NOT EXISTS trading.positions (
  account_id    bigint NOT NULL REFERENCES trading.accounts (account_id) ON DELETE CASCADE,
  symbol        text NOT NULL CHECK (symbol ~ '^[A-Z0-9.-]{1,10}$'),
  name          text CHECK (name IS NULL OR char_length(name) BETWEEN 1 AND 200),
  shares        numeric(18, 6) NOT NULL CHECK (shares > 0),
  cost_basis    numeric(14, 4) NOT NULL CHECK (cost_basis > 0),
  acquired      date,
  recent_shares numeric(18, 6) CHECK (recent_shares IS NULL OR (recent_shares > 0 AND recent_shares <= shares)),
  PRIMARY KEY (account_id, symbol)
);

CREATE TABLE IF NOT EXISTS trading.trades (
  trade_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id    bigint NOT NULL REFERENCES trading.accounts (account_id) ON DELETE CASCADE,
  symbol        text NOT NULL CHECK (symbol ~ '^[A-Z0-9.-]{1,10}$'),
  side          text NOT NULL CHECK (side IN ('buy', 'sell')),
  shares        numeric(18, 6) NOT NULL CHECK (shares > 0),
  price         numeric(14, 4) NOT NULL CHECK (price > 0),
  realized_gain numeric(14, 2) NOT NULL DEFAULT 0,
  traded_on     date NOT NULL,
  record_id     text,
  source        text NOT NULL CHECK (source IN ('user', 'bot', 'seed')),
  created_at    timestamptz NOT NULL DEFAULT now(),
  CHECK (side <> 'buy' OR realized_gain = 0)
);

COMMENT ON COLUMN trading.trades.realized_gain IS
  'A sell: round(price * shares, 2) minus round(cost basis * shares, 2), in dollars. A buy is 0. A deposit is not a gain.';

CREATE INDEX IF NOT EXISTS trades_account_day ON trading.trades (account_id, traded_on);

CREATE TABLE IF NOT EXISTS trading.recommendation_decisions (
  decision_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id  bigint NOT NULL REFERENCES trading.accounts (account_id) ON DELETE CASCADE,
  record_id   text NOT NULL CHECK (char_length(record_id) BETWEEN 1 AND 80),
  symbol      text NOT NULL CHECK (symbol ~ '^[A-Z0-9.-]{1,10}$'),
  action      text NOT NULL CHECK (action ~ '^[a-z_]{1,32}$'),
  shares      numeric(18, 6) NOT NULL CHECK (shares >= 0),
  mode        text NOT NULL CHECK (mode IN ('ask', 'auto')),
  will_trade  text NOT NULL CHECK (will_trade IN (
    'yes_auto', 'yes_user', 'pending_user',
    'no_system_denied', 'no_user_denied', 'no_expired', 'no_funds')),
  reason      text NOT NULL,
  decided_on  date NOT NULL,
  CONSTRAINT recommendation_one_per_record UNIQUE (account_id, record_id)
);

CREATE INDEX IF NOT EXISTS decisions_account_day ON trading.recommendation_decisions (account_id, decided_on);

-- Design choices entry 110 (Jacob, 2026-11-15): a share quantity may have a fraction (an even split buys the same DOLLARS of each stock), at most six decimals (prometheus/common/shares.py: PLACES). An account book made before
-- that has these four columns as integer: change them to numeric(18, 6) (idempotent: a column that already has the type is left as it is; every whole number keeps its value, and no table or account is dropped). The CHECKs stay as
-- they are. No trigger, function or view holds a share count.
ALTER TABLE trading.positions
  ALTER COLUMN shares TYPE numeric(18, 6),
  ALTER COLUMN recent_shares TYPE numeric(18, 6);
ALTER TABLE trading.trades
  ALTER COLUMN shares TYPE numeric(18, 6);
ALTER TABLE trading.recommendation_decisions
  ALTER COLUMN shares TYPE numeric(18, 6);

-- Design choices entry 140 (Jacob, 2026-11-21): a direct order whose limit price is not met WAITS as a DAY order on the paper broker, with its cash (a purchase) or its shares (a sale) SET ASIDE. One row for each order.
-- `reserved_cash` and `reserved_shares` are what the order holds while its status is `open`: the account's AVAILABLE cash is the cash less the reserved cash of its open orders, and the shares it may sell are the
-- shares held less the reserved shares (AccountStore.available_cash / available_shares, the one door every reader of what may be spent or sold goes through). A purchase holds its most possible cost (the shares typed
-- times the limit, or the dollars of an order in dollars), a sale holds its shares. An order is placed, filled, expired and cancelled in ONE transaction that locks the account row first (the same lock every booking takes),
-- so a placement cannot race a fill or another placement. This is additive: CREATE TABLE IF NOT EXISTS keeps every table and account.
CREATE TABLE IF NOT EXISTS trading.open_orders (
  order_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id      bigint NOT NULL REFERENCES trading.accounts (account_id) ON DELETE CASCADE,
  request_id      text NOT NULL CHECK (char_length(request_id) BETWEEN 1 AND 80),
  record_id       text NOT NULL CHECK (char_length(record_id) BETWEEN 1 AND 80),
  symbol          text NOT NULL CHECK (symbol ~ '^[A-Z0-9.-]{1,10}$'),
  company         text CHECK (company IS NULL OR char_length(company) BETWEEN 1 AND 200),
  side            text NOT NULL CHECK (side IN ('buy', 'sell')),
  shares          numeric(18, 6) CHECK (shares IS NULL OR shares > 0),
  dollars         numeric(14, 2) CHECK (dollars IS NULL OR dollars > 0),
  limit_price     numeric(14, 4) NOT NULL CHECK (limit_price > 0),
  reserved_cash   numeric(14, 2) NOT NULL DEFAULT 0 CHECK (reserved_cash >= 0),
  reserved_shares numeric(18, 6) NOT NULL DEFAULT 0 CHECK (reserved_shares >= 0),
  request         text NOT NULL,
  placed_at       timestamptz NOT NULL DEFAULT now(),
  expires_at      timestamptz NOT NULL,
  status          text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'filled', 'expired', 'cancelled', 'rejected')),
  closed_at       timestamptz,
  reason          text,
  trade_id        bigint,
  raise_funds_by  text,
  CONSTRAINT open_order_shares_or_dollars CHECK ((shares IS NOT NULL) <> (dollars IS NOT NULL)),
  CONSTRAINT open_order_holds_one_thing CHECK ((side = 'buy' AND reserved_shares = 0 AND (reserved_cash > 0 OR raise_funds_by IS NOT NULL)) OR (side = 'sell' AND reserved_cash = 0 AND reserved_shares > 0 AND dollars IS NULL AND raise_funds_by IS NULL)),
  CONSTRAINT open_order_closed_has_a_time CHECK (status = 'open' OR closed_at IS NOT NULL),
  CONSTRAINT open_order_one_per_record UNIQUE (account_id, record_id)
);

-- Design choices entry 143 (a waiting PURCHASE whose cash is short, with the user's permission to raise funds): `raise_funds_by` says how the user permitted selling holdings to pay for it WHEN THE PRICE IS REACHED ('words': their own words, 'setting':
-- ALWAYS_RAISE_FUNDS); nothing is sold while it waits. Such an order sets aside only the cash that is free (it may be none), so a purchase may hold 0 when `raise_funds_by` is set. For a book that already has the table (entry 140), additive and
-- safe to run again: the column is added if it is missing and the one constraint on what an order holds is replaced.
ALTER TABLE trading.open_orders ADD COLUMN IF NOT EXISTS raise_funds_by text;
ALTER TABLE trading.open_orders DROP CONSTRAINT IF EXISTS open_order_raise_funds_by_known;
ALTER TABLE trading.open_orders ADD CONSTRAINT open_order_raise_funds_by_known CHECK (raise_funds_by IS NULL OR raise_funds_by IN ('words', 'setting'));
ALTER TABLE trading.open_orders DROP CONSTRAINT IF EXISTS open_order_holds_one_thing;
ALTER TABLE trading.open_orders ADD CONSTRAINT open_order_holds_one_thing CHECK ((side = 'buy' AND reserved_shares = 0 AND (reserved_cash > 0 OR raise_funds_by IS NOT NULL)) OR (side = 'sell' AND reserved_cash = 0 AND reserved_shares > 0 AND dollars IS NULL AND raise_funds_by IS NULL));

COMMENT ON TABLE trading.open_orders IS
  'A direct order that waits for its limit price (a DAY order, paper broker only). While status is open it SETS ASIDE reserved_cash (a purchase) or reserved_shares (a sale): that money and those shares are not available to anything else.';

CREATE INDEX IF NOT EXISTS open_orders_open ON trading.open_orders (account_id) WHERE status = 'open';

-- The watcher in the web server checks the orders of EVERY account, and the application role sees only the account it names (row-level security). These two functions are the two questions it may ask across accounts: which
-- accounts have an open order, and which account an order belongs to (a cancel by order number). Neither returns anything else.
CREATE OR REPLACE FUNCTION trading.accounts_with_open_orders() RETURNS TABLE (account_id bigint)
LANGUAGE sql
SECURITY DEFINER
SET search_path = trading, pg_temp
STABLE
AS $$
  SELECT DISTINCT o.account_id FROM open_orders o WHERE o.status = 'open' ORDER BY o.account_id
$$;

CREATE OR REPLACE FUNCTION trading.account_of_open_order(p_order_id bigint) RETURNS bigint
LANGUAGE sql
SECURITY DEFINER
SET search_path = trading, pg_temp
STABLE
AS $$
  SELECT o.account_id FROM open_orders o WHERE o.order_id = p_order_id
$$;

CREATE OR REPLACE FUNCTION trading.create_account(
  p_display_name text,
  p_risk_tolerance text,
  p_preferred_name text,
  p_email text,
  p_phone text,
  p_timezone text,
  p_trade_mode text,
  p_note text,
  p_age integer,
  p_horizon numeric
) RETURNS bigint
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = trading, pg_temp
AS $$
DECLARE
  new_id bigint;
BEGIN
  INSERT INTO accounts (
    display_name, risk_tolerance, preferred_name, email, phone, timezone, trade_mode, note, age, investment_horizon_years
  ) VALUES (
    p_display_name, p_risk_tolerance, p_preferred_name, p_email, p_phone, p_timezone, p_trade_mode, p_note, p_age, p_horizon
  )
  RETURNING account_id INTO new_id;
  RETURN new_id;
END;
$$;

-- Sample rows (seed_key set) are replaced only by sql/seed_test_users.sql, which runs as the superuser.
CREATE OR REPLACE FUNCTION trading.delete_account(p_account_id bigint) RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = trading, pg_temp
AS $$
DECLARE
  removed integer;
BEGIN
  DELETE FROM accounts WHERE account_id = p_account_id AND seed_key IS NULL;
  GET DIAGNOSTICS removed = ROW_COUNT;
  IF removed <> 1 THEN
    RAISE EXCEPTION 'that account cannot be deleted';
  END IF;
END;
$$;

CREATE OR REPLACE FUNCTION trading.account_id_for_seed(p_seed_key text) RETURNS bigint
LANGUAGE sql
SECURITY DEFINER
SET search_path = trading, pg_temp
STABLE
AS $$
  SELECT account_id FROM accounts WHERE seed_key = p_seed_key
$$;

DROP POLICY IF EXISTS account_isolation ON trading.accounts;
CREATE POLICY account_isolation ON trading.accounts
  USING (account_id = trading.current_account_id())
  WITH CHECK (account_id = trading.current_account_id());

DROP POLICY IF EXISTS ledger_isolation ON trading.cash_ledger;
CREATE POLICY ledger_isolation ON trading.cash_ledger
  USING (account_id = trading.current_account_id())
  WITH CHECK (account_id = trading.current_account_id());

DROP POLICY IF EXISTS position_isolation ON trading.positions;
CREATE POLICY position_isolation ON trading.positions
  USING (account_id = trading.current_account_id())
  WITH CHECK (account_id = trading.current_account_id());

DROP POLICY IF EXISTS trade_isolation ON trading.trades;
CREATE POLICY trade_isolation ON trading.trades
  USING (account_id = trading.current_account_id())
  WITH CHECK (account_id = trading.current_account_id());

DROP POLICY IF EXISTS open_order_isolation ON trading.open_orders;
CREATE POLICY open_order_isolation ON trading.open_orders
  USING (account_id = trading.current_account_id())
  WITH CHECK (account_id = trading.current_account_id());

DROP POLICY IF EXISTS decision_isolation ON trading.recommendation_decisions;
CREATE POLICY decision_isolation ON trading.recommendation_decisions
  USING (account_id = trading.current_account_id())
  WITH CHECK (account_id = trading.current_account_id());

ALTER TABLE trading.accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading.accounts FORCE ROW LEVEL SECURITY;
ALTER TABLE trading.cash_ledger ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading.cash_ledger FORCE ROW LEVEL SECURITY;
ALTER TABLE trading.positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading.positions FORCE ROW LEVEL SECURITY;
ALTER TABLE trading.trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading.trades FORCE ROW LEVEL SECURITY;
ALTER TABLE trading.open_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading.open_orders FORCE ROW LEVEL SECURITY;
ALTER TABLE trading.recommendation_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading.recommendation_decisions FORCE ROW LEVEL SECURITY;

RESET ROLE;

REVOKE ALL ON SCHEMA trading FROM PUBLIC;
GRANT USAGE ON SCHEMA trading TO prometheus_app;
REVOKE ALL ON ALL TABLES IN SCHEMA trading FROM PUBLIC;
GRANT SELECT ON ALL TABLES IN SCHEMA trading TO prometheus_app;
GRANT INSERT ON trading.cash_ledger, trading.trades, trading.positions, trading.recommendation_decisions TO prometheus_app;
GRANT UPDATE, DELETE ON trading.positions TO prometheus_app;
GRANT UPDATE ON trading.recommendation_decisions TO prometheus_app;
GRANT INSERT, UPDATE ON trading.open_orders TO prometheus_app;
GRANT UPDATE (
  display_name, preferred_name, email, phone, timezone, status, risk_tolerance, trade_mode,
  age, investment_horizon_years, note, updated_at
) ON trading.accounts TO prometheus_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA trading TO prometheus_app;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA trading FROM PUBLIC;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA trading TO prometheus_app;

COMMIT;
