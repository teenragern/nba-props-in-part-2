from src.models.edge_ranker import rank_edges


def _candidate(player='Player A', market='player_points', side='OVER',
               model_prob=0.60, implied_prob=0.50, odds=2.0,
               proj_minutes=28.0, mean=22.0, line=20.5,
               injury_status='Healthy', variance_scale=1.0,
               steam_flag=False, velocity=0.0, dispersion=0.02, book_role='rec'):
    return dict(
        player_id=player, market=market, side=side,
        model_prob=model_prob, implied_prob=implied_prob, odds=odds,
        projected_minutes=proj_minutes, mean=mean, line=line,
        injury_status=injury_status, variance_scale=variance_scale,
        steam_flag=steam_flag, velocity=velocity, dispersion=dispersion,
        book_role=book_role, book='draftkings',
    )


def test_returns_sorted_by_risk_adjusted_ev():
    low  = _candidate(model_prob=0.52, player='Low Edge')
    high = _candidate(model_prob=0.65, player='High Edge')
    ranked = rank_edges([low, high])
    assert ranked[0]['player_id'] == 'High Edge', "Higher edge should rank first"


def test_out_player_excluded():
    c = _candidate(injury_status='Out')
    ranked = rank_edges([c])
    assert len(ranked) == 0, "OUT player should be excluded from ranking"


def test_low_minutes_excluded():
    c = _candidate(proj_minutes=10.0)  # below MIN_PROJECTED_MINUTES=15
    ranked = rank_edges([c])
    assert len(ranked) == 0, "Low-minutes player should be filtered out"


def test_steam_boosts_edge():
    no_steam   = _candidate(steam_flag=False, player='No Steam')
    with_steam = _candidate(steam_flag=True,  player='Steam')
    ranked = rank_edges([no_steam, with_steam])
    steam_result   = next(r for r in ranked if r['player_id'] == 'Steam')
    nosteam_result = next(r for r in ranked if r['player_id'] == 'No Steam')
    assert steam_result['edge'] > nosteam_result['edge'], "Steam should boost edge"


def test_doubtful_reduces_edge():
    healthy  = _candidate(injury_status='Healthy',  player='Healthy')
    doubtful = _candidate(injury_status='Doubtful', player='Doubtful')
    ranked = rank_edges([healthy, doubtful])
    h_edge = next(r for r in ranked if r['player_id'] == 'Healthy')['edge']
    d_edge = next(r for r in ranked if r['player_id'] == 'Doubtful')['edge']
    assert d_edge < h_edge, "Doubtful should have smaller edge than Healthy"


def test_empty_candidates_returns_empty():
    assert rank_edges([]) == []


def test_sharp_book_role_boosts_edge():
    rec   = _candidate(book_role='rec',   player='Rec')
    sharp = _candidate(book_role='sharp', player='Sharp')
    ranked = rank_edges([rec, sharp])
    sharp_edge = next(r for r in ranked if r['player_id'] == 'Sharp')['edge']
    rec_edge   = next(r for r in ranked if r['player_id'] == 'Rec')['edge']
    assert sharp_edge > rec_edge, "Sharp book should receive edge boost"
