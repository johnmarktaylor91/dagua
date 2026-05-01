# Perceptual Divergence Report

Ranks use Tier A Agg cards. Rank 1 is worst for both L1 and SSIM_loss.

- Mean Agg SSIM: 0.963315
- Mean Cairo SSIM: 0.963242

## Largest L1-vs-SSIM Divergences

| Card | L1 | SSIM_loss | L1 rank | SSIM_loss rank | Divergence |
| --- | ---: | ---: | ---: | ---: | ---: |
| evil_donut_diamond | 2.024 | 0.019939 | 31 | 134 | 103 |
| nodes_fills_gradient_radial | 2.108 | 0.025099 | 30 | 96 | 66 |
| evil_pie_star | 1.128 | 0.016579 | 85 | 151 | 66 |
| combo_parallelogram_dotted | 1.307 | 0.064960 | 69 | 10 | 59 |
| combo_star_dotted | 1.315 | 0.064555 | 67 | 11 | 56 |
| evil_long_wrap_star | 0.413 | 0.028256 | 145 | 95 | 50 |
| nodes_text_font_style_italic | 0.397 | 0.022946 | 154 | 105 | 49 |
| edges_styles_style_dotted | 0.414 | 0.024111 | 143 | 100 | 43 |
| combo_dashed_diamond_opacity | 1.463 | 0.060639 | 59 | 16 | 43 |
| combo_cloud_gradient_italic | 2.501 | 0.049104 | 25 | 68 | 43 |
| combo_diamond_dashed_opacity_italic | 1.454 | 0.060398 | 60 | 18 | 42 |
| nodes_text_font_weight_regular | 0.408 | 0.022842 | 147 | 107 | 40 |
| nodes_text_font_style_normal | 0.408 | 0.022842 | 148 | 108 | 40 |
| nodes_fills_fill_pattern_solid | 0.408 | 0.022842 | 149 | 109 | 40 |
| edges_styles_style_dashed | 0.401 | 0.022705 | 152 | 112 | 40 |
| edges_styles_width_0_5 | 0.397 | 0.022634 | 153 | 113 | 40 |
| clusters_opacity_1_0 | 1.542 | 0.029000 | 53 | 93 | 40 |
| combo_box3d_gradient | 2.805 | 0.051456 | 22 | 62 | 40 |
| nodes_shapes_rect | 0.525 | 0.016414 | 115 | 154 | 39 |
| nodes_shapes_cylinder | 0.576 | 0.016646 | 111 | 150 | 39 |

## L1-Blind Candidates

Cards where L1 ranks the card relatively good but SSIM_loss ranks it worse.

| Card | L1 | SSIM_loss | L1 rank | SSIM_loss rank | Rank gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| combo_parallelogram_dotted | 1.307 | 0.064960 | 69 | 10 | 59 |
| combo_star_dotted | 1.315 | 0.064555 | 67 | 11 | 56 |
| evil_long_wrap_star | 0.413 | 0.028256 | 145 | 95 | 50 |
| nodes_text_font_style_italic | 0.397 | 0.022946 | 154 | 105 | 49 |
| edges_styles_style_dotted | 0.414 | 0.024111 | 143 | 100 | 43 |
| combo_dashed_diamond_opacity | 1.463 | 0.060639 | 59 | 16 | 43 |
| combo_diamond_dashed_opacity_italic | 1.454 | 0.060398 | 60 | 18 | 42 |
| nodes_text_font_weight_regular | 0.408 | 0.022842 | 147 | 107 | 40 |
| nodes_text_font_style_normal | 0.408 | 0.022842 | 148 | 108 | 40 |
| nodes_fills_fill_pattern_solid | 0.408 | 0.022842 | 149 | 109 | 40 |

## Metric-Noise Candidates

Cards where L1 ranks the card worse but SSIM_loss ranks it relatively good.

| Card | L1 | SSIM_loss | L1 rank | SSIM_loss rank | Rank gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| evil_donut_diamond | 2.024 | 0.019939 | 31 | 134 | 103 |
| nodes_fills_gradient_radial | 2.108 | 0.025099 | 30 | 96 | 66 |
| evil_pie_star | 1.128 | 0.016579 | 85 | 151 | 66 |
| combo_cloud_gradient_italic | 2.501 | 0.049104 | 25 | 68 | 43 |
| clusters_opacity_1_0 | 1.542 | 0.029000 | 53 | 93 | 40 |
| combo_box3d_gradient | 2.805 | 0.051456 | 22 | 62 | 40 |
| nodes_shapes_rect | 0.525 | 0.016414 | 115 | 154 | 39 |
| nodes_shapes_cylinder | 0.576 | 0.016646 | 111 | 150 | 39 |
| combo_gradient_rounded | 3.226 | 0.055342 | 13 | 52 | 39 |
| graph_background_near_black | 0.903 | 0.020349 | 94 | 131 | 37 |

## Top Cairo SSIM Wins

| Card | Agg SSIM | Cairo SSIM | SSIM delta | L1 delta |
| --- | ---: | ---: | ---: | ---: |
| nodes_text_text_valign_center | 0.954120 | 0.956110 | 0.001990 | -0.395 |
| nodes_text_text_valign_top | 0.956256 | 0.958221 | 0.001965 | -0.395 |
| combo_bt_cluster_rounded | 0.950109 | 0.951068 | 0.000960 | -0.070 |
| edges_arrows_circle | 0.979487 | 0.980382 | 0.000896 | -0.011 |
| edges_arrows_open | 0.979428 | 0.980261 | 0.000833 | -0.012 |
| combo_cluster_rounded_gradient | 0.950515 | 0.951344 | 0.000829 | -0.087 |
| edges_arrows_vee | 0.979455 | 0.980279 | 0.000824 | -0.012 |
| edges_arrows_arrow_fill_hollow | 0.979244 | 0.980057 | 0.000813 | -0.012 |
| edges_arrows_arrow_fill_filled | 0.979493 | 0.980290 | 0.000797 | -0.010 |
| edges_arrows_normal | 0.979493 | 0.980290 | 0.000797 | -0.010 |

## Smoking-Gun Card

- Card: `clusters_stroke_dash_dashed`
- Agg L1: 0.857
- Cairo L1: 0.823
- L1 delta (Cairo - Agg): -0.034
- Agg SSIM_loss: 0.032892
- Cairo SSIM_loss: 0.032733
- SSIM_loss delta (Cairo - Agg): -0.000159
