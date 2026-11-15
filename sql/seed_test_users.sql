-- Three sample accounts for the account book (design choices entry 72).
-- Re-running deletes those three and inserts them again, with new account numbers.
-- Any other account is left alone. Run this as the database superuser (scripts/setup_postgres.sh).

BEGIN;

DELETE FROM trading.accounts WHERE seed_key IS NOT NULL;

DO $$
DECLARE
  jacob bigint;
  avery bigint;
  blake bigint;
  cash numeric(14, 2);
  gain numeric(14, 2);
BEGIN
  INSERT INTO trading.accounts (seed_key, display_name, risk_tolerance, trade_mode, note)
  VALUES (
    'jacob_sample', 'Jacob Pyrett', 'moderate', 'ask',
    'Sample account for the JSON portfolio and profile. The holdings were already there, so they are not trades and they are not part of the gain. Not a real brokerage account.'
  ) RETURNING account_id INTO jacob;

  INSERT INTO trading.cash_ledger (account_id, kind, amount, balance_after, memo, occurred_on)
  VALUES (jacob, 'deposit', 51.23, 51.23, 'opening cash', DATE '2024-01-02');

  INSERT INTO trading.positions (account_id, symbol, name, shares, cost_basis, acquired) VALUES
    (jacob, 'GM', 'General Motors Co', 60, 10.0000, DATE '2023-09-05'),
    (jacob, 'MSFT', 'Microsoft Corp', 50, 150.0000, DATE '2024-03-15'),
    (jacob, 'TSLA', 'Tesla Inc', 123, 300.0000, DATE '2024-06-20');

  INSERT INTO trading.recommendation_decisions
    (account_id, record_id, symbol, action, shares, mode, will_trade, reason, decided_on)
  VALUES
    (jacob, 'rec_jacob_mu', 'MU', 'add', 2, 'ask', 'yes_user', 'the user approved', DATE '2026-11-03'),
    (jacob, 'rec_jacob_closed', 'GM', 'add', 10, 'ask', 'no_system_denied', 'the market is closed', DATE '2026-11-03');

  INSERT INTO trading.accounts (seed_key, display_name, preferred_name, email, risk_tolerance, trade_mode, note)
  VALUES (
    'avery_chen', 'Avery Chen', 'Avery', 'avery.chen@example.com', 'conservative', 'ask',
    'Made-up account. One deposit, two buys, two sells. Realized gain for 2026 is -27.50.'
  ) RETURNING account_id INTO avery;

  INSERT INTO trading.cash_ledger (account_id, kind, amount, balance_after, memo, occurred_on)
  VALUES (avery, 'deposit', 5000.00, 5000.00, 'deposit', DATE '2026-01-15');
  INSERT INTO trading.cash_ledger (account_id, kind, amount, balance_after, memo, occurred_on)
  VALUES (avery, 'buy', -250.00, 4750.00, 'buy 20 F', DATE '2026-02-02');
  INSERT INTO trading.cash_ledger (account_id, kind, amount, balance_after, memo, occurred_on)
  VALUES (avery, 'buy', -400.00, 4350.00, 'buy 10 GM', DATE '2026-03-10');
  INSERT INTO trading.cash_ledger (account_id, kind, amount, balance_after, memo, occurred_on)
  VALUES (avery, 'sell', 75.00, 4425.00, 'sell 5 F', DATE '2026-05-20');
  INSERT INTO trading.cash_ledger (account_id, kind, amount, balance_after, memo, occurred_on)
  VALUES (avery, 'sell', 360.00, 4785.00, 'sell 10 GM', DATE '2026-07-01');

  INSERT INTO trading.trades (account_id, symbol, side, shares, price, realized_gain, traded_on, source) VALUES
    (avery, 'F', 'buy', 20, 12.5000, 0, DATE '2026-02-02', 'seed'),
    (avery, 'GM', 'buy', 10, 40.0000, 0, DATE '2026-03-10', 'seed'),
    (avery, 'F', 'sell', 5, 15.0000, 12.50, DATE '2026-05-20', 'seed'),
    (avery, 'GM', 'sell', 10, 36.0000, -40.00, DATE '2026-07-01', 'seed');

  INSERT INTO trading.positions (account_id, symbol, name, shares, cost_basis, acquired)
  VALUES (avery, 'F', 'Ford Motor Co', 15, 12.5000, DATE '2026-02-02');

  INSERT INTO trading.recommendation_decisions
    (account_id, record_id, symbol, action, shares, mode, will_trade, reason, decided_on)
  VALUES
    (avery, 'rec_avery_buy_f', 'F', 'add', 20, 'ask', 'yes_user', 'the user approved', DATE '2026-03-01'),
    (avery, 'rec_avery_denied', 'GM', 'add', 10, 'ask', 'no_system_denied', 'the review denied it', DATE '2026-04-01'),
    (avery, 'rec_avery_sell_f', 'F', 'reduce', 5, 'auto', 'yes_auto', 'the score cleared the automatic mark', DATE '2026-05-01'),
    (avery, 'rec_avery_pending', 'F', 'add', 1, 'ask', 'pending_user', 'waiting for the user', DATE '2026-08-01');

  INSERT INTO trading.accounts (seed_key, display_name, preferred_name, email, risk_tolerance, trade_mode, note)
  VALUES (
    'blake_okonkwo', 'Blake Okonkwo', 'Blake', 'blake.okonkwo@example.com', 'aggressive', 'auto',
    'Made-up account. One deposit, one buy, no sells, two denied recommendations.'
  ) RETURNING account_id INTO blake;

  INSERT INTO trading.cash_ledger (account_id, kind, amount, balance_after, memo, occurred_on)
  VALUES (blake, 'deposit', 2000.00, 2000.00, 'deposit', DATE '2026-04-01');
  INSERT INTO trading.cash_ledger (account_id, kind, amount, balance_after, memo, occurred_on)
  VALUES (blake, 'buy', -200.00, 1800.00, 'buy 8 PFE', DATE '2026-04-02');

  INSERT INTO trading.trades (account_id, symbol, side, shares, price, realized_gain, traded_on, source)
  VALUES (blake, 'PFE', 'buy', 8, 25.0000, 0, DATE '2026-04-02', 'seed');

  INSERT INTO trading.positions (account_id, symbol, name, shares, cost_basis, acquired)
  VALUES (blake, 'PFE', 'Pfizer Inc', 8, 25.0000, DATE '2026-04-02');

  INSERT INTO trading.recommendation_decisions
    (account_id, record_id, symbol, action, shares, mode, will_trade, reason, decided_on)
  VALUES
    (blake, 'rec_blake_funds', 'PFE', 'add', 100, 'auto', 'no_funds', 'the cash does not cover it', DATE '2026-06-15'),
    (blake, 'rec_blake_user', 'PFE', 'reduce', 1, 'auto', 'no_user_denied', 'the user denied', DATE '2026-06-16');

  SELECT cash_balance INTO cash FROM trading.accounts WHERE account_id = jacob;
  IF cash <> 51.23 THEN
    RAISE EXCEPTION 'Jacob cash is %, expected 51.23', cash;
  END IF;
  SELECT cash_balance INTO cash FROM trading.accounts WHERE account_id = avery;
  IF cash <> 4785.00 THEN
    RAISE EXCEPTION 'Avery cash is %, expected 4785.00', cash;
  END IF;
  SELECT cash_balance INTO cash FROM trading.accounts WHERE account_id = blake;
  IF cash <> 1800.00 THEN
    RAISE EXCEPTION 'Blake cash is %, expected 1800.00', cash;
  END IF;
  SELECT COALESCE(SUM(realized_gain), 0) INTO gain FROM trading.trades WHERE account_id = avery;
  IF gain <> -27.50 THEN
    RAISE EXCEPTION 'Avery realized gain is %, expected -27.50', gain;
  END IF;
  IF (SELECT count(*) FROM trading.positions WHERE account_id = jacob) <> 3 THEN
    RAISE EXCEPTION 'Jacob should hold 3 positions';
  END IF;
  IF (SELECT shares FROM trading.positions WHERE account_id = avery AND symbol = 'F') <> 15 THEN
    RAISE EXCEPTION 'Avery should hold 15 F';
  END IF;
  IF (SELECT count(*) FROM trading.trades WHERE account_id = avery) <> 4 THEN
    RAISE EXCEPTION 'Avery should have 4 trades';
  END IF;
  IF (SELECT count(*) FROM trading.recommendation_decisions WHERE account_id = blake AND will_trade LIKE 'no_%') <> 2 THEN
    RAISE EXCEPTION 'Blake should have 2 denials';
  END IF;
END
$$;

COMMIT;
