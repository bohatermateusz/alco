package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.surface_view.LineDrawView5;
import com.hismith.hismithapp.view.RightDraggableFrameLayout;
import master.flame.danmaku.ui.widget.DanmakuView;

/* loaded from: classes3.dex */
public final class ActivityPlayModePlay2Binding implements ViewBinding {
    public final ImageView imgAutoPlay;
    public final ImageView imgModePlayPiu;
    public final ImageView imgSettingPosition;
    public final ImageView imgSettingPositionArrow;
    public final ImageView imgSettingSmooth;
    public final RightDraggableFrameLayout layoutFragmentPiu;
    public final LinearLayout layoutMark;
    public final ConstraintLayout layoutSetting;
    public final DanmakuView modePlayDanmaku;
    public final EditText modePlayEditBarrageContent;
    public final ImageView modePlayImgBarrage;
    public final ImageView modePlayImgBarrageSend;
    public final ImageView modePlayImgCollectCount;
    public final ImageView modePlayImgLikeCount;
    public final ImageView modePlayImgMore;
    public final ImageView modePlayImgPlay;
    public final ImageView modePlayImgPlayCount;
    public final ImageView modePlayImgType;
    public final ConstraintLayout modePlayLayout;
    public final ConstraintLayout modePlayLayoutAction;
    public final ConstraintLayout modePlayLayoutBarrage;
    public final FrameLayout modePlayLayoutBarrageColor;
    public final LinearLayout modePlayLayoutBarrageF;
    public final ConstraintLayout modePlayLayoutTime;
    public final LineDrawView5 modePlayLineDrawView;
    public final RecyclerView modePlayRecyclerColor;
    public final SeekBar modePlaySeekBar;
    public final AppCompatTextView modePlayTxtCollectCount;
    public final AppCompatTextView modePlayTxtEndTime;
    public final AppCompatTextView modePlayTxtLikeCount;
    public final AppCompatTextView modePlayTxtPlayCount;
    public final AppCompatTextView modePlayTxtStartTime;
    public final View modePlayViewBarrageColor;
    private final ConstraintLayout rootView;
    public final TitleBarBinding titleLayout;
    public final TextView txtModeDepth;
    public final TextView txtModeMaxStroke;
    public final TextView txtModeSmooth;
    public final TextView txtModeSpeed;
    public final AppCompatTextView txtSettingPositionMax;
    public final AppCompatTextView txtSettingPositionMin;
    public final TextView txtSettingSmooth;
    public final View viewHeight;

    private ActivityPlayModePlay2Binding(ConstraintLayout constraintLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, RightDraggableFrameLayout rightDraggableFrameLayout, LinearLayout linearLayout, ConstraintLayout constraintLayout2, DanmakuView danmakuView, EditText editText, ImageView imageView6, ImageView imageView7, ImageView imageView8, ImageView imageView9, ImageView imageView10, ImageView imageView11, ImageView imageView12, ImageView imageView13, ConstraintLayout constraintLayout3, ConstraintLayout constraintLayout4, ConstraintLayout constraintLayout5, FrameLayout frameLayout, LinearLayout linearLayout2, ConstraintLayout constraintLayout6, LineDrawView5 lineDrawView5, RecyclerView recyclerView, SeekBar seekBar, AppCompatTextView appCompatTextView, AppCompatTextView appCompatTextView2, AppCompatTextView appCompatTextView3, AppCompatTextView appCompatTextView4, AppCompatTextView appCompatTextView5, View view, TitleBarBinding titleBarBinding, TextView textView, TextView textView2, TextView textView3, TextView textView4, AppCompatTextView appCompatTextView6, AppCompatTextView appCompatTextView7, TextView textView5, View view2) {
        this.rootView = constraintLayout;
        this.imgAutoPlay = imageView;
        this.imgModePlayPiu = imageView2;
        this.imgSettingPosition = imageView3;
        this.imgSettingPositionArrow = imageView4;
        this.imgSettingSmooth = imageView5;
        this.layoutFragmentPiu = rightDraggableFrameLayout;
        this.layoutMark = linearLayout;
        this.layoutSetting = constraintLayout2;
        this.modePlayDanmaku = danmakuView;
        this.modePlayEditBarrageContent = editText;
        this.modePlayImgBarrage = imageView6;
        this.modePlayImgBarrageSend = imageView7;
        this.modePlayImgCollectCount = imageView8;
        this.modePlayImgLikeCount = imageView9;
        this.modePlayImgMore = imageView10;
        this.modePlayImgPlay = imageView11;
        this.modePlayImgPlayCount = imageView12;
        this.modePlayImgType = imageView13;
        this.modePlayLayout = constraintLayout3;
        this.modePlayLayoutAction = constraintLayout4;
        this.modePlayLayoutBarrage = constraintLayout5;
        this.modePlayLayoutBarrageColor = frameLayout;
        this.modePlayLayoutBarrageF = linearLayout2;
        this.modePlayLayoutTime = constraintLayout6;
        this.modePlayLineDrawView = lineDrawView5;
        this.modePlayRecyclerColor = recyclerView;
        this.modePlaySeekBar = seekBar;
        this.modePlayTxtCollectCount = appCompatTextView;
        this.modePlayTxtEndTime = appCompatTextView2;
        this.modePlayTxtLikeCount = appCompatTextView3;
        this.modePlayTxtPlayCount = appCompatTextView4;
        this.modePlayTxtStartTime = appCompatTextView5;
        this.modePlayViewBarrageColor = view;
        this.titleLayout = titleBarBinding;
        this.txtModeDepth = textView;
        this.txtModeMaxStroke = textView2;
        this.txtModeSmooth = textView3;
        this.txtModeSpeed = textView4;
        this.txtSettingPositionMax = appCompatTextView6;
        this.txtSettingPositionMin = appCompatTextView7;
        this.txtSettingSmooth = textView5;
        this.viewHeight = view2;
    }

    public static ActivityPlayModePlay2Binding bind(View view) {
        View findChildViewById;
        View findChildViewById2;
        View findChildViewById3;
        int i = R.id.img_auto_play;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_mode_play_piu;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.img_setting_position;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.img_setting_position_arrow;
                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView4 != null) {
                        i = R.id.img_setting_smooth;
                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView5 != null) {
                            i = R.id.layout_fragment_piu;
                            RightDraggableFrameLayout rightDraggableFrameLayout = (RightDraggableFrameLayout) ViewBindings.findChildViewById(view, i);
                            if (rightDraggableFrameLayout != null) {
                                i = R.id.layout_mark;
                                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout != null) {
                                    i = R.id.layout_setting;
                                    ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                    if (constraintLayout != null) {
                                        i = R.id.mode_play_danmaku;
                                        DanmakuView danmakuView = (DanmakuView) ViewBindings.findChildViewById(view, i);
                                        if (danmakuView != null) {
                                            i = R.id.mode_play_edit_barrage_content;
                                            EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                                            if (editText != null) {
                                                i = R.id.mode_play_img_barrage;
                                                ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                if (imageView6 != null) {
                                                    i = R.id.mode_play_img_barrage_send;
                                                    ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                    if (imageView7 != null) {
                                                        i = R.id.mode_play_img_collectCount;
                                                        ImageView imageView8 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                        if (imageView8 != null) {
                                                            i = R.id.mode_play_img_likeCount;
                                                            ImageView imageView9 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                            if (imageView9 != null) {
                                                                i = R.id.mode_play_img_more;
                                                                ImageView imageView10 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                if (imageView10 != null) {
                                                                    i = R.id.mode_play_img_play;
                                                                    ImageView imageView11 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                    if (imageView11 != null) {
                                                                        i = R.id.mode_play_img_playCount;
                                                                        ImageView imageView12 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                        if (imageView12 != null) {
                                                                            i = R.id.mode_play_img_type;
                                                                            ImageView imageView13 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                            if (imageView13 != null) {
                                                                                i = R.id.mode_play_layout;
                                                                                ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                                if (constraintLayout2 != null) {
                                                                                    i = R.id.mode_play_layout_action;
                                                                                    ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                                    if (constraintLayout3 != null) {
                                                                                        i = R.id.mode_play_layout_barrage;
                                                                                        ConstraintLayout constraintLayout4 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                                        if (constraintLayout4 != null) {
                                                                                            i = R.id.mode_play_layout_barrage_color;
                                                                                            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                            if (frameLayout != null) {
                                                                                                i = R.id.mode_play_layout_barrage_f;
                                                                                                LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                                if (linearLayout2 != null) {
                                                                                                    i = R.id.mode_play_layout_time;
                                                                                                    ConstraintLayout constraintLayout5 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                                                    if (constraintLayout5 != null) {
                                                                                                        i = R.id.mode_play_lineDrawView;
                                                                                                        LineDrawView5 lineDrawView5 = (LineDrawView5) ViewBindings.findChildViewById(view, i);
                                                                                                        if (lineDrawView5 != null) {
                                                                                                            i = R.id.mode_play_recycler_color;
                                                                                                            RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                                            if (recyclerView != null) {
                                                                                                                i = R.id.mode_play_seek_bar;
                                                                                                                SeekBar seekBar = (SeekBar) ViewBindings.findChildViewById(view, i);
                                                                                                                if (seekBar != null) {
                                                                                                                    i = R.id.mode_play_txt_collectCount;
                                                                                                                    AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                    if (appCompatTextView != null) {
                                                                                                                        i = R.id.mode_play_txt_end_time;
                                                                                                                        AppCompatTextView appCompatTextView2 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                        if (appCompatTextView2 != null) {
                                                                                                                            i = R.id.mode_play_txt_likeCount;
                                                                                                                            AppCompatTextView appCompatTextView3 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                            if (appCompatTextView3 != null) {
                                                                                                                                i = R.id.mode_play_txt_playCount;
                                                                                                                                AppCompatTextView appCompatTextView4 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                if (appCompatTextView4 != null) {
                                                                                                                                    i = R.id.mode_play_txt_start_time;
                                                                                                                                    AppCompatTextView appCompatTextView5 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                    if (appCompatTextView5 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.mode_play_view_barrage_color))) != null && (findChildViewById2 = ViewBindings.findChildViewById(view, (i = R.id.title_layout))) != null) {
                                                                                                                                        TitleBarBinding bind = TitleBarBinding.bind(findChildViewById2);
                                                                                                                                        i = R.id.txt_mode_depth;
                                                                                                                                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                        if (textView != null) {
                                                                                                                                            i = R.id.txt_mode_max_stroke;
                                                                                                                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                            if (textView2 != null) {
                                                                                                                                                i = R.id.txt_mode_smooth;
                                                                                                                                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                if (textView3 != null) {
                                                                                                                                                    i = R.id.txt_mode_speed;
                                                                                                                                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                    if (textView4 != null) {
                                                                                                                                                        i = R.id.txt_setting_position_max;
                                                                                                                                                        AppCompatTextView appCompatTextView6 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                        if (appCompatTextView6 != null) {
                                                                                                                                                            i = R.id.txt_setting_position_min;
                                                                                                                                                            AppCompatTextView appCompatTextView7 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                            if (appCompatTextView7 != null) {
                                                                                                                                                                i = R.id.txt_setting_smooth;
                                                                                                                                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                if (textView5 != null && (findChildViewById3 = ViewBindings.findChildViewById(view, (i = R.id.view_height))) != null) {
                                                                                                                                                                    return new ActivityPlayModePlay2Binding((ConstraintLayout) view, imageView, imageView2, imageView3, imageView4, imageView5, rightDraggableFrameLayout, linearLayout, constraintLayout, danmakuView, editText, imageView6, imageView7, imageView8, imageView9, imageView10, imageView11, imageView12, imageView13, constraintLayout2, constraintLayout3, constraintLayout4, frameLayout, linearLayout2, constraintLayout5, lineDrawView5, recyclerView, seekBar, appCompatTextView, appCompatTextView2, appCompatTextView3, appCompatTextView4, appCompatTextView5, findChildViewById, bind, textView, textView2, textView3, textView4, appCompatTextView6, appCompatTextView7, textView5, findChildViewById3);
                                                                                                                                                                }
                                                                                                                                                            }
                                                                                                                                                        }
                                                                                                                                                    }
                                                                                                                                                }
                                                                                                                                            }
                                                                                                                                        }
                                                                                                                                    }
                                                                                                                                }
                                                                                                                            }
                                                                                                                        }
                                                                                                                    }
                                                                                                                }
                                                                                                            }
                                                                                                        }
                                                                                                    }
                                                                                                }
                                                                                            }
                                                                                        }
                                                                                    }
                                                                                }
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityPlayModePlay2Binding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPlayModePlay2Binding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_play_mode_play2, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public ConstraintLayout getRoot() {
        return this.rootView;
    }
}
