/*
 * lc_core2.c  —  Lost Cities: complete game logic in C
 *
 * Design
 * ──────
 * • All hot game state lives in a C struct (LCGame).
 *   Python never touches the inner arrays directly.
 * • Cards encoded as a single uint8: color*10 + value_idx
 *   (value_idx = 0 for investment, v-1 for numbered cards v∈{2..10})
 * • get_valid_actions writes flat action ints directly — no Python tuple
 *   allocation, no enum attribute access.
 * • apply_play / apply_draw / take_action fully in C.
 * • build_obs and fill_action_mask work directly from the struct.
 * • Python only drives: deck shuffle (random.Random), opponent.act_batch(),
 *   and the outer step() loop that calls these C functions.
 *
 * Action encoding  (total 600)
 * ─────────────────────────────
 *   flat = card_id * 12 + action_type * 6 + draw_source
 *   card_id     ∈ [0,50)   card_pool = 5 colors × 10 unique values
 *   action_type ∈ {0=Play, 1=Discard}
 *   draw_source ∈ [0,6)    0=deck, 1..5=discard pile index+1
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* ── Constants ────────────────────────────────────────────────────────── */
#define NUM_COLORS      5
#define UNIQUE_VALS     10
#define CARD_POOL       50      /* NUM_COLORS * UNIQUE_VALS                */
#define ACTION_PLAY     0
#define ACTION_DISCARD  1
#define NUM_DRAW_SRC    6       /* 0=deck, 1..5=discard pile               */
#define TOTAL_ACTIONS   600     /* CARD_POOL * 2 * NUM_DRAW_SRC            */
#define MAX_HAND        10      /* 8 in practice + 2 safety                */
#define MAX_EXP         14      /* max 12 cards (3 inv + 9 num) + 2 safety */
#define MAX_DISCARD     60
#define DECK_SIZE       60
#define NUM_PLAYERS     2
#define STARTING_HAND   8
#define OBS_SIZE        301     /* 6*CARD_POOL + 1                         */

/* ── Card encoding ────────────────────────────────────────────────────── */
static inline uint8_t card_encode(int color, int value) {
    return (uint8_t)(color * UNIQUE_VALS + (value == 0 ? 0 : value - 1));
}
static inline int card_color(uint8_t c)      { return c / UNIQUE_VALS; }
static inline int card_value_idx(uint8_t c)  { return c % UNIQUE_VALS; }
static inline int card_value(uint8_t c)      { int v = c % UNIQUE_VALS; return v + (v != 0); }

/* ── Action encoding ─────────────────────────────────────────────────── */
static inline int action_flat(int card_id, int atype, int draw_src) {
    return card_id * (2 * NUM_DRAW_SRC) + atype * NUM_DRAW_SRC + draw_src;
}
static inline void action_unflatten(int flat, int *card_id, int *atype, int *draw_src) {
    *draw_src = flat % NUM_DRAW_SRC; flat /= NUM_DRAW_SRC;
    *atype    = flat % 2;            *card_id = flat / 2;
}

/* ── Game struct ─────────────────────────────────────────────────────── */
typedef struct {
    /* deck as a stack; deck[0..deck_top-1] are active, deck[deck_top] is next pop */
    uint8_t deck[DECK_SIZE];
    int     deck_top;           /* number of cards remaining               */

    uint8_t hands[NUM_PLAYERS][MAX_HAND];
    int     hand_size[NUM_PLAYERS];

    /* expeditions[player][color]: cards in order played                   */
    uint8_t expeditions[NUM_PLAYERS][NUM_COLORS][MAX_EXP];
    int     exp_size[NUM_PLAYERS][NUM_COLORS];

    /* discard piles: discard[color][0..dp_size-1], top is dp_size-1       */
    uint8_t discard[NUM_COLORS][MAX_DISCARD];
    int     dp_size[NUM_COLORS];

    int     current_player;
    int     game_over;
    float   scores[NUM_PLAYERS];
} LCGame;

/* ── Validity ─────────────────────────────────────────────────────────── */
static inline int valid_expedition_play(const LCGame *g, int player, int color, int value) {
    int sz = g->exp_size[player][color];
    if (sz == 0) return 1;
    const uint8_t *exp = g->expeditions[player][color];
    int last_value = card_value(exp[sz - 1]);
    if (value == 0) {
        /* count existing investments */
        int inv = 0;
        for (int i = 0; i < sz; i++) if (card_value(exp[i]) == 0) inv++;
        if (inv >= 3) return 0;
        return last_value == 0;
    }
    return value > (last_value != 0 ? last_value : 0);
}

/* ── Scoring ─────────────────────────────────────────────────────────── */
static float score_player(const LCGame *g, int player) {
    float total = 0.0f;
    for (int c = 0; c < NUM_COLORS; c++) {
        int sz = g->exp_size[player][c];
        if (sz == 0) continue;
        const uint8_t *exp = g->expeditions[player][c];
        int base = 0, inv = 0;
        for (int i = 0; i < sz; i++) {
            int v = card_value(exp[i]);
            base += v;
            if (v == 0) inv++;
        }
        int score = (base - 20) * (inv + 1);
        if (sz >= 8) score += 20;
        total += score;
    }
    return total;
}

static void update_scores(LCGame *g) {
    g->scores[0] = score_player(g, 0);
    g->scores[1] = score_player(g, 1);
}

/* ── Core game functions ─────────────────────────────────────────────── */

/*
 * get_valid_actions_c
 * Writes flat action indices into out_buf (caller must provide ≥128 ints).
 * Returns count written.
 */
int get_valid_actions_c(const LCGame *g, int player, int *out_buf) {
    int n = 0;
    int hand_size = g->hand_size[player];
    const uint8_t *hand = g->hands[player];

    for (int h = 0; h < hand_size; h++) {
        uint8_t card    = hand[h];
        int color       = card_color(card);
        int value       = card_value(card);
        int card_id     = (int)card;   /* card byte IS the card_id (0..49) */
        int can_play    = valid_expedition_play(g, player, color, value);

        for (int ds = 0; ds < NUM_DRAW_SRC; ds++) {
            /* ds 0 = deck (available if deck non-empty — checked by caller) */
            if (ds > 0 && g->dp_size[ds - 1] == 0) continue;
            if (can_play)
                out_buf[n++] = action_flat(card_id, ACTION_PLAY, ds);
            /* discard: can't draw from the pile we just discarded to */
            if (ds == 0 || (ds - 1) != color)
                out_buf[n++] = action_flat(card_id, ACTION_DISCARD, ds);
        }
    }
    return n;
}

/*
 * fill_action_mask_c
 * Sets mask[flat] = 1 for each valid action. mask must be pre-zeroed (len TOTAL_ACTIONS).
 */
void fill_action_mask_c(const LCGame *g, int player, uint8_t *mask) {
    int buf[128]; int n = get_valid_actions_c(g, player, buf);
    for (int i = 0; i < n; i++) mask[buf[i]] = 1;
}

/*
 * apply_play_c
 * Removes card from hand; places on expedition or discard pile.
 * Returns 0 on success, -1 on error.
 */
int apply_play_c(LCGame *g, int player, uint8_t card, int action_type) {
    /* find and remove from hand */
    int *hs = &g->hand_size[player];
    uint8_t *hand = g->hands[player];
    int found = -1;
    for (int i = 0; i < *hs; i++) {
        if (hand[i] == card) { found = i; break; }
    }
    if (found < 0) return -1;
    /* swap-remove */
    hand[found] = hand[--(*hs)];

    int color = card_color(card);
    int value = card_value(card);

    if (action_type == ACTION_PLAY) {
        if (!valid_expedition_play(g, player, color, value)) return -1;
        int sz = g->exp_size[player][color];
        g->expeditions[player][color][sz] = card;
        g->exp_size[player][color]++;
    } else {
        int sz = g->dp_size[color];
        g->discard[color][sz] = card;
        g->dp_size[color]++;
    }
    return 0;
}

/*
 * apply_draw_c
 * Draws a replacement card.
 * draw_source: -1 = deck, 0..4 = discard pile index.
 * Also advances current_player and sets game_over when deck exhausts.
 * Returns 0 on success, -1 on error.
 */
int apply_draw_c(LCGame *g, int player, int draw_source) {
    uint8_t drawn;
    if (draw_source == -1) {
        if (g->deck_top == 0) return -1;
        drawn = g->deck[--g->deck_top];
    } else {
        if (g->dp_size[draw_source] == 0) return -1;
        drawn = g->discard[draw_source][--g->dp_size[draw_source]];
    }
    g->hands[player][g->hand_size[player]++] = drawn;

    if (g->deck_top == 0) {
        g->game_over = 1;
        update_scores(g);
    } else {
        g->current_player = 1 - player;
    }
    return 0;
}

/*
 * take_action_c — combined play+draw (backward-compatible single-turn API).
 * draw_source_env: 0 = deck, 1..5 = discard pile index+1 (env convention).
 */
int take_action_c(LCGame *g, int player, uint8_t card,
                  int action_type, int draw_source_env) {
    if (g->current_player != player) return -2;
    int draw_source = draw_source_env == 0 ? -1 : draw_source_env - 1;
    int color = card_color(card);
    if (draw_source >= 0) {
        if (g->dp_size[draw_source] == 0) return -3;
        if (action_type == ACTION_DISCARD && draw_source == color) return -4;
    }
    if (apply_play_c(g, player, card, action_type) != 0) return -5;
    if (apply_draw_c(g, player, draw_source) != 0) return -6;
    return 0;
}

/* ── Observation builder ─────────────────────────────────────────────── */
/*
 * Layout (OBS_SIZE = 301):
 *   [0:50]   hand counts
 *   [50:100] own expedition counts
 *   [100:150] opp expedition counts
 *   [150:200] discard top-1 counts
 *   [200:250] discard top-2 counts
 *   [250:300] discard top-3 counts
 *   [300]    deck_size (clamped to int8 range)
 */
void build_obs_c(const LCGame *g, int player, int8_t *obs) {
    memset(obs, 0, OBS_SIZE);
    int opp = 1 - player;

    /* hand */
    int hs = g->hand_size[player];
    const uint8_t *hand = g->hands[player];
    for (int i = 0; i < hs; i++) obs[(int)hand[i]] += 1;

    /* own expeditions */
    for (int c = 0; c < NUM_COLORS; c++) {
        int sz = g->exp_size[player][c];
        const uint8_t *exp = g->expeditions[player][c];
        for (int i = 0; i < sz; i++) obs[50 + (int)exp[i]] += 1;
    }

    /* opponent expeditions */
    for (int c = 0; c < NUM_COLORS; c++) {
        int sz = g->exp_size[opp][c];
        const uint8_t *exp = g->expeditions[opp][c];
        for (int i = 0; i < sz; i++) obs[100 + (int)exp[i]] += 1;
    }

    /* discard top-3 layers */
    for (int layer = 0; layer < 3; layer++) {
        int base = 150 + layer * CARD_POOL;
        for (int c = 0; c < NUM_COLORS; c++) {
            int sz = g->dp_size[c];
            if (sz > layer)
                obs[base + (int)g->discard[c][sz - layer - 1]] += 1;
        }
    }

    obs[300] = (int8_t)(g->deck_top > 127 ? 127 : g->deck_top);
}

/* ── Batch helpers (for VecEnv) ──────────────────────────────────────── */

/*
 * collect_obs_masks_batch
 * For N games (pointer array), fill obs_batch[N*OBS_SIZE] and mask_batch[N*TOTAL_ACTIONS].
 * Row-major layout.
 */
void collect_obs_masks_batch(
        const LCGame **games, int N, int player,
        int8_t  *obs_batch,    /* N × OBS_SIZE     */
        uint8_t *mask_batch)   /* N × TOTAL_ACTIONS */
{
    for (int i = 0; i < N; i++) {
        int8_t  *obs  = obs_batch  + i * OBS_SIZE;
        uint8_t *mask = mask_batch + i * TOTAL_ACTIONS;
        memset(mask, 0, TOTAL_ACTIONS);
        build_obs_c(games[i], player, obs);
        fill_action_mask_c(games[i], player, mask);
    }
}

/*
 * scores_batch
 * Fill scores[N*2] for N games.
 */
void scores_batch(const LCGame **games, int N, float *scores) {
    for (int i = 0; i < N; i++) {
        scores[i * 2 + 0] = score_player(games[i], 0);
        scores[i * 2 + 1] = score_player(games[i], 1);
    }
}

/* ── Reset helper ────────────────────────────────────────────────────── */
/*
 * init_game_from_deck
 * Called after Python shuffles the deck.
 * deck_cards: 60 uint8 encoded cards (already shuffled), index 0 = first drawn.
 * Deals STARTING_HAND cards to each player (alternating, last 8 pairs from deck).
 */
void init_game_from_deck(LCGame *g, const uint8_t *deck_cards, int n_cards) {
    /* clear all state */
    memset(g, 0, sizeof(LCGame));
    g->game_over = 0;
    g->current_player = 0;

    /* copy deck (we'll pop from the end — index deck_top-1 is top) */
    g->deck_top = n_cards;
    memcpy(g->deck, deck_cards, n_cards);

    /* deal starting hands: alternating p0, p1 as in original */
    for (int i = 0; i < STARTING_HAND; i++) {
        for (int p = 0; p < NUM_PLAYERS; p++) {
            g->hands[p][g->hand_size[p]++] = g->deck[--g->deck_top];
        }
    }
}

/* ── Accessors for Python ─────────────────────────────────────────────── */
int    lc2_deck_top(const LCGame *g)         { return g->deck_top; }
int    lc2_current_player(const LCGame *g)   { return g->current_player; }
int    lc2_game_over(const LCGame *g)        { return g->game_over; }
float  lc2_score(const LCGame *g, int p)     { return g->scores[p]; }
int    lc2_hand_size(const LCGame *g, int p) { return g->hand_size[p]; }
uint8_t lc2_hand_card(const LCGame *g, int p, int i) { return g->hands[p][i]; }
int    lc2_exp_size(const LCGame *g, int p, int c) { return g->exp_size[p][c]; }
uint8_t lc2_exp_card(const LCGame *g, int p, int c, int i) { return g->expeditions[p][c][i]; }
int    lc2_dp_size(const LCGame *g, int c)   { return g->dp_size[c]; }
uint8_t lc2_dp_top(const LCGame *g, int c)   { return g->discard[c][g->dp_size[c]-1]; }
size_t lc2_sizeof_game(void)                 { return sizeof(LCGame); }

/* ── Test helpers: set game state directly ────────────────────────────── */
void lc2_set_expedition(LCGame *g, int player, int color,
                        const uint8_t *cards, int n) {
    g->exp_size[player][color] = n;
    for (int i = 0; i < n; i++) g->expeditions[player][color][i] = cards[i];
}
void lc2_set_hand(LCGame *g, int player, const uint8_t *cards, int n) {
    g->hand_size[player] = n;
    for (int i = 0; i < n; i++) g->hands[player][i] = cards[i];
}
void lc2_update_scores(LCGame *g) { update_scores(g); }

/* ── Constant accessors ───────────────────────────────────────────────── */
int lc2_total_actions(void) { return TOTAL_ACTIONS; }
int lc2_card_pool(void)     { return CARD_POOL; }
int lc2_obs_size(void)      { return OBS_SIZE; }
int lc2_num_colors(void)    { return NUM_COLORS; }

/* ── Action encode/decode for Python ─────────────────────────────────── */
int lc2_action_flat(int card_id, int atype, int draw_src) {
    return action_flat(card_id, atype, draw_src);
}
void lc2_action_unflatten(int flat, int *card_id, int *atype, int *draw_src) {
    action_unflatten(flat, card_id, atype, draw_src);
}
int lc2_card_color(uint8_t c)     { return card_color(c); }
int lc2_card_value(uint8_t c)     { return card_value(c); }
uint8_t lc2_card_encode(int color, int value) { return card_encode(color, value); }
