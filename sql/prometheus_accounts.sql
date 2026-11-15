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
  auto_min_score            integer CHECK (auto_min_score IS NULL OR auto_min_score BETWEEN 0 AND 100),
  max_position_pct          numeric(6, 4) NOT NULL DEFAULT 0.1000 CHECK (max_position_pct > 0 AND max_position_pct <= 1),
  max_loss_per_trade_pct    numeric(6, 4) NOT NULL DEFAULT 0.0100 CHECK (max_loss_per_trade_pct > 0 AND max_loss_per_trade_pct <= 1),
  max_new_exposure_pct      numeric(6, 4) NOT NULL DEFAULT 0.3000 CHECK (max_new_exposure_pct > 0 AND max_new_exposure_pct <= 1),
  max_order_dollars         numeric(14, 2) CHECK (max_order_dollars IS NULL OR max_order_dollars > 0),
  min_price                 numeric(14, 4) NOT NULL DEFAULT 5 CHECK (min_price >= 0),
  excluded_symbols          text[] NOT NULL DEFAULT '{}',
  excluded_sectors          text[] NOT NULL DEFAULT '{}',
  note                      text CHECK (note IS NULL OR char_length(note) <= 2000),
  cash_balance              numeric(14, 2) NOT NULL DEFAULT 0 CHECK (cash_balance >= 0),
  created_at                timestamptz NOT NULL DEFAULT now(),
  updated_at                timestamptz NOT NULL DEFAULT now()
);

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
  shares        integer NOT NULL CHECK (shares > 0),
  cost_basis    numeric(14, 4) NOT NULL CHECK (cost_basis > 0),
  acquired      date,
  recent_shares integer CHECK (recent_shares IS NULL OR (recent_shares > 0 AND recent_shares <= shares)),
  PRIMARY KEY (account_id, symbol)
);

CREATE TABLE IF NOT EXISTS trading.trades (
  trade_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  account_id    bigint NOT NULL REFERENCES trading.accounts (account_id) ON DELETE CASCADE,
  symbol        text NOT NULL CHECK (symbol ~ '^[A-Z0-9.-]{1,10}$'),
  side          text NOT NULL CHECK (side IN ('buy', 'sell')),
  shares        integer NOT NULL CHECK (shares > 0),
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
  shares      integer NOT NULL CHECK (shares >= 0),
  mode        text NOT NULL CHECK (mode IN ('ask', 'auto')),
  will_trade  text NOT NULL CHECK (will_trade IN (
    'yes_auto', 'yes_user', 'pending_user',
    'no_system_denied', 'no_user_denied', 'no_expired', 'no_funds')),
  reason      text NOT NULL,
  decided_on  date NOT NULL,
  CONSTRAINT recommendation_one_per_record UNIQUE (account_id, record_id)
);

CREATE INDEX IF NOT EXISTS decisions_account_day ON trading.recommendation_decisions (account_id, decided_on);

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
GRANT UPDATE (
  display_name, preferred_name, email, phone, timezone, status, risk_tolerance, trade_mode,
  age, investment_horizon_years, note, updated_at
) ON trading.accounts TO prometheus_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA trading TO prometheus_app;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA trading FROM PUBLIC;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA trading TO prometheus_app;

COMMIT;
