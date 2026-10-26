package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.surface_view.LineDrawView5;
import com.hismith.hismithapp.view.BothWayArrowView;
import com.hismith.hismithapp.view.DoubleSliderView;
import com.hismith.hismithapp.view.NodeSliderView;
import com.hismith.hismithapp.view.PlaySettingImageView;
import com.hismith.hismithapp.view.TouchLayout;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;

/* loaded from: classes3.dex */
public final class ActivityPlayModeCreate2Binding implements ViewBinding {
    public final BothWayArrowView arrowBoth;
    public final TextView btnSave;
    public final ConstraintLayout drawerLayout;
    public final ImageView imgGesture;
    public final ImageView imgModeCopy;
    public final ImageView imgModeDel;
    public final ImageView imgModeList;
    public final ImageView imgModePaste;
    public final ImageView imgModePlay;
    public final ImageView imgModeSelect;
    public final PlaySettingImageView imgModeSetting;
    public final ImageView imgModeStart;
    public final ImageView imgModeStop;
    public final ImageView imgPiu;
    public final ImageView imgReversal;
    public final ImageView imgTouch;
    public final ImageView imgTouch1;
    public final LinearLayout layoutAction;
    public final LinearLayout layoutArrow;
    public final FrameLayout layoutBg;
    public final FrameLayout layoutBg1;
    public final FrameLayout layoutDraw;
    public final LinearLayout layoutMark;
    public final LinearLayout layoutModeList;
    public final FrameLayout layoutTime;
    public final FrameLayout layoutTouch;
    public final LineDrawView5 lineDrawView5;
    public final RecyclerView recyclerMode;
    public final SmartRefreshLayout refreshLayoutMode;
    private final ConstraintLayout rootView;
    public final SeekBar seekTime;
    public final DoubleSliderView sliderPosition;
    public final NodeSliderView sliderSmooth;
    public final ImageView titleImgBluetooth;
    public final LinearLayout titleLayout;
    public final View titleLine;
    public final TextView titleTxtName;
    public final TouchLayout touchLayout;
    public final TouchLayout touchLayout1;
    public final TextView txtEndTime;
    public final TextView txtModeDepth;
    public final TextView txtModeList;
    public final TextView txtModeMaxStroke;
    public final TextView txtModeSmooth;
    public final TextView txtModeSpeed;
    public final TextView txtStartTime;

    private ActivityPlayModeCreate2Binding(ConstraintLayout constraintLayout, BothWayArrowView bothWayArrowView, TextView textView, ConstraintLayout constraintLayout2, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, PlaySettingImageView playSettingImageView, ImageView imageView8, ImageView imageView9, ImageView imageView10, ImageView imageView11, ImageView imageView12, ImageView imageView13, LinearLayout linearLayout, LinearLayout linearLayout2, FrameLayout frameLayout, FrameLayout frameLayout2, FrameLayout frameLayout3, LinearLayout linearLayout3, LinearLayout linearLayout4, FrameLayout frameLayout4, FrameLayout frameLayout5, LineDrawView5 lineDrawView5, RecyclerView recyclerView, SmartRefreshLayout smartRefreshLayout, SeekBar seekBar, DoubleSliderView doubleSliderView, NodeSliderView nodeSliderView, ImageView imageView14, LinearLayout linearLayout5, View view, TextView textView2, TouchLayout touchLayout, TouchLayout touchLayout2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9) {
        this.rootView = constraintLayout;
        this.arrowBoth = bothWayArrowView;
        this.btnSave = textView;
        this.drawerLayout = constraintLayout2;
        this.imgGesture = imageView;
        this.imgModeCopy = imageView2;
        this.imgModeDel = imageView3;
        this.imgModeList = imageView4;
        this.imgModePaste = imageView5;
        this.imgModePlay = imageView6;
        this.imgModeSelect = imageView7;
        this.imgModeSetting = playSettingImageView;
        this.imgModeStart = imageView8;
        this.imgModeStop = imageView9;
        this.imgPiu = imageView10;
        this.imgReversal = imageView11;
        this.imgTouch = imageView12;
        this.imgTouch1 = imageView13;
        this.layoutAction = linearLayout;
        this.layoutArrow = linearLayout2;
        this.layoutBg = frameLayout;
        this.layoutBg1 = frameLayout2;
        this.layoutDraw = frameLayout3;
        this.layoutMark = linearLayout3;
        this.layoutModeList = linearLayout4;
        this.layoutTime = frameLayout4;
        this.layoutTouch = frameLayout5;
        this.lineDrawView5 = lineDrawView5;
        this.recyclerMode = recyclerView;
        this.refreshLayoutMode = smartRefreshLayout;
        this.seekTime = seekBar;
        this.sliderPosition = doubleSliderView;
        this.sliderSmooth = nodeSliderView;
        this.titleImgBluetooth = imageView14;
        this.titleLayout = linearLayout5;
        this.titleLine = view;
        this.titleTxtName = textView2;
        this.touchLayout = touchLayout;
        this.touchLayout1 = touchLayout2;
        this.txtEndTime = textView3;
        this.txtModeDepth = textView4;
        this.txtModeList = textView5;
        this.txtModeMaxStroke = textView6;
        this.txtModeSmooth = textView7;
        this.txtModeSpeed = textView8;
        this.txtStartTime = textView9;
    }

    public static ActivityPlayModeCreate2Binding bind(View view) {
        View findChildViewById;
        int i = R.id.arrow_both;
        BothWayArrowView bothWayArrowView = (BothWayArrowView) ViewBindings.findChildViewById(view, i);
        if (bothWayArrowView != null) {
            i = R.id.btn_save;
            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView != null) {
                ConstraintLayout constraintLayout = (ConstraintLayout) view;
                i = R.id.img_gesture;
                ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView != null) {
                    i = R.id.img_mode_copy;
                    ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView2 != null) {
                        i = R.id.img_mode_del;
                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView3 != null) {
                            i = R.id.img_mode_list;
                            ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView4 != null) {
                                i = R.id.img_mode_paste;
                                ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView5 != null) {
                                    i = R.id.img_mode_play;
                                    ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView6 != null) {
                                        i = R.id.img_mode_select;
                                        ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView7 != null) {
                                            i = R.id.img_mode_setting;
                                            PlaySettingImageView playSettingImageView = (PlaySettingImageView) ViewBindings.findChildViewById(view, i);
                                            if (playSettingImageView != null) {
                                                i = R.id.img_mode_start;
                                                ImageView imageView8 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                if (imageView8 != null) {
                                                    i = R.id.img_mode_stop;
                                                    ImageView imageView9 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                    if (imageView9 != null) {
                                                        i = R.id.img_piu;
                                                        ImageView imageView10 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                        if (imageView10 != null) {
                                                            i = R.id.img_reversal;
                                                            ImageView imageView11 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                            if (imageView11 != null) {
                                                                i = R.id.img_touch;
                                                                ImageView imageView12 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                if (imageView12 != null) {
                                                                    i = R.id.img_touch1;
                                                                    ImageView imageView13 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                    if (imageView13 != null) {
                                                                        i = R.id.layout_action;
                                                                        LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                        if (linearLayout != null) {
                                                                            i = R.id.layout_arrow;
                                                                            LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                            if (linearLayout2 != null) {
                                                                                i = R.id.layout_bg;
                                                                                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                if (frameLayout != null) {
                                                                                    i = R.id.layout_bg1;
                                                                                    FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                    if (frameLayout2 != null) {
                                                                                        i = R.id.layout_draw;
                                                                                        FrameLayout frameLayout3 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                        if (frameLayout3 != null) {
                                                                                            i = R.id.layout_mark;
                                                                                            LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                            if (linearLayout3 != null) {
                                                                                                i = R.id.layout_mode_list;
                                                                                                LinearLayout linearLayout4 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                                if (linearLayout4 != null) {
                                                                                                    i = R.id.layout_time;
                                                                                                    FrameLayout frameLayout4 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                    if (frameLayout4 != null) {
                                                                                                        i = R.id.layout_touch;
                                                                                                        FrameLayout frameLayout5 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                        if (frameLayout5 != null) {
                                                                                                            i = R.id.lineDrawView5;
                                                                                                            LineDrawView5 lineDrawView5 = (LineDrawView5) ViewBindings.findChildViewById(view, i);
                                                                                                            if (lineDrawView5 != null) {
                                                                                                                i = R.id.recycler_mode;
                                                                                                                RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                                                if (recyclerView != null) {
                                                                                                                    i = R.id.refreshLayout_mode;
                                                                                                                    SmartRefreshLayout smartRefreshLayout = (SmartRefreshLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                    if (smartRefreshLayout != null) {
                                                                                                                        i = R.id.seek_time;
                                                                                                                        SeekBar seekBar = (SeekBar) ViewBindings.findChildViewById(view, i);
                                                                                                                        if (seekBar != null) {
                                                                                                                            i = R.id.slider_position;
                                                                                                                            DoubleSliderView doubleSliderView = (DoubleSliderView) ViewBindings.findChildViewById(view, i);
                                                                                                                            if (doubleSliderView != null) {
                                                                                                                                i = R.id.slider_smooth;
                                                                                                                                NodeSliderView nodeSliderView = (NodeSliderView) ViewBindings.findChildViewById(view, i);
                                                                                                                                if (nodeSliderView != null) {
                                                                                                                                    i = R.id.title_img_bluetooth;
                                                                                                                                    ImageView imageView14 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                                                                    if (imageView14 != null) {
                                                                                                                                        i = R.id.title_layout;
                                                                                                                                        LinearLayout linearLayout5 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                                        if (linearLayout5 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.title_line))) != null) {
                                                                                                                                            i = R.id.title_txt_name;
                                                                                                                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                            if (textView2 != null) {
                                                                                                                                                i = R.id.touch_layout;
                                                                                                                                                TouchLayout touchLayout = (TouchLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                                                if (touchLayout != null) {
                                                                                                                                                    i = R.id.touch_layout1;
                                                                                                                                                    TouchLayout touchLayout2 = (TouchLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                                                    if (touchLayout2 != null) {
                                                                                                                                                        i = R.id.txt_end_time;
                                                                                                                                                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                        if (textView3 != null) {
                                                                                                                                                            i = R.id.txt_mode_depth;
                                                                                                                                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                            if (textView4 != null) {
                                                                                                                                                                i = R.id.txt_mode_list;
                                                                                                                                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                if (textView5 != null) {
                                                                                                                                                                    i = R.id.txt_mode_max_stroke;
                                                                                                                                                                    TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                    if (textView6 != null) {
                                                                                                                                                                        i = R.id.txt_mode_smooth;
                                                                                                                                                                        TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                        if (textView7 != null) {
                                                                                                                                                                            i = R.id.txt_mode_speed;
                                                                                                                                                                            TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                            if (textView8 != null) {
                                                                                                                                                                                i = R.id.txt_start_time;
                                                                                                                                                                                TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                if (textView9 != null) {
                                                                                                                                                                                    return new ActivityPlayModeCreate2Binding(constraintLayout, bothWayArrowView, textView, constraintLayout, imageView, imageView2, imageView3, imageView4, imageView5, imageView6, imageView7, playSettingImageView, imageView8, imageView9, imageView10, imageView11, imageView12, imageView13, linearLayout, linearLayout2, frameLayout, frameLayout2, frameLayout3, linearLayout3, linearLayout4, frameLayout4, frameLayout5, lineDrawView5, recyclerView, smartRefreshLayout, seekBar, doubleSliderView, nodeSliderView, imageView14, linearLayout5, findChildViewById, textView2, touchLayout, touchLayout2, textView3, textView4, textView5, textView6, textView7, textView8, textView9);
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
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityPlayModeCreate2Binding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPlayModeCreate2Binding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_play_mode_create2, viewGroup, false);
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
