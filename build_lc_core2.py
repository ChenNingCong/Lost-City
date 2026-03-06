"""
build_lc_core2.py — compile lc_core2.c (complete game engine in C).
Run once: python3 build_lc_core2.py
"""
import cffi, os

ffi = cffi.FFI()

ffi.cdef("""
/* opaque game struct */
typedef struct LCGame LCGame;

/* constants */
int    lc2_total_actions(void);
int    lc2_card_pool(void);
int    lc2_obs_size(void);
int    lc2_num_colors(void);
size_t lc2_sizeof_game(void);

/* card encode/decode */
uint8_t lc2_card_encode(int color, int value);
int     lc2_card_color(uint8_t c);
int     lc2_card_value(uint8_t c);

/* action encode/decode */
int  lc2_action_flat(int card_id, int atype, int draw_src);
void lc2_action_unflatten(int flat, int *card_id, int *atype, int *draw_src);

/* game init */
void init_game_from_deck(LCGame *g, const uint8_t *deck_cards, int n_cards);

/* game actions */
int get_valid_actions_c(const LCGame *g, int player, int *out_buf);
void fill_action_mask_c(const LCGame *g, int player, uint8_t *mask);
int  apply_play_c(LCGame *g, int player, uint8_t card, int action_type);
int  apply_draw_c(LCGame *g, int player, int draw_source);
int  take_action_c(LCGame *g, int player, uint8_t card,
                   int action_type, int draw_source_env);

/* observation */
void build_obs_c(const LCGame *g, int player, int8_t *obs);

/* batch (vec env) */
void collect_obs_masks_batch(
    const LCGame **games, int N, int player,
    int8_t  *obs_batch,
    uint8_t *mask_batch);
void scores_batch(const LCGame **games, int N, float *scores);

/* accessors */
int     lc2_deck_top(const LCGame *g);
int     lc2_current_player(const LCGame *g);
int     lc2_game_over(const LCGame *g);
float   lc2_score(const LCGame *g, int p);
int     lc2_hand_size(const LCGame *g, int p);
uint8_t lc2_hand_card(const LCGame *g, int p, int i);
int     lc2_exp_size(const LCGame *g, int p, int c);
uint8_t lc2_exp_card(const LCGame *g, int p, int c, int i);
int     lc2_dp_size(const LCGame *g, int c);
uint8_t lc2_dp_top(const LCGame *g, int c);

/* test helpers */
void lc2_set_expedition(LCGame *g, int player, int color,
                        const uint8_t *cards, int n);
void lc2_set_hand(LCGame *g, int player, const uint8_t *cards, int n);
void lc2_update_scores(LCGame *g);
""")

with open(os.path.join(os.path.dirname(__file__), "lc_core2.c")) as f:
    source = f.read()

ffi.set_source(
    "lc_core2_cffi",
    source,
    extra_compile_args=["-O3", "-march=native", "-ffast-math", "-std=c11"],
)

if __name__ == "__main__":
    ffi.compile(tmpdir=".", verbose=True)
    print("Build complete: lc_core2_cffi")
