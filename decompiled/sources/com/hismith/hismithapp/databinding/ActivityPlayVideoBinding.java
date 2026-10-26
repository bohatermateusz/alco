package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Group;
import androidx.media3.ui.PlayerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.pose.OverlayView;

/* loaded from: classes3.dex */
public final class ActivityPlayVideoBinding implements ViewBinding {
    public final View bgSelectVideo;
    public final ProgressBar cachingSpinner;
    public final Group groupEmpty;
    public final Group groupPlaying;
    public final ImageView imgBluetooth;
    public final ImageView imgPanelGesture;
    public final ImageView imgSelectVideo;
    public final ImageView imgVideoEmpty;
    public final ImageView imgVideoLoop;
    public final ImageView imgVideoNext;
    public final ImageView imgVideoPanel;
    public final ImageView imgVideoPlay;
    public final ImageView imgVideoPrevious;
    public final ImageView imgVideoSelect;
    public final LinearLayout layoutCachingHint;
    public final ConstraintLayout layoutControl;
    public final ConstraintLayout layoutSpeed;
    public final LinearLayout layoutTitle;
    public final ConstraintLayout main;
    public final OverlayView overlay;
    public final PlayerView playView;
    private final ConstraintLayout rootView;
    public final SeekBar seekVideo;
    public final ImageView titleImgBack;
    public final TextView titleTxtName;
    public final TextView txtCachingHint;
    public final TextView txtSelectVideo;
    public final TextView txtSpeed;
    public final TextView txtVideoEmpty;
    public final TextView txtVideoTimeCurrent;
    public final TextView txtVideoTimeTotal;
    public final View viewHeight;

    private ActivityPlayVideoBinding(ConstraintLayout constraintLayout, View view, ProgressBar progressBar, Group group, Group group2, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, ImageView imageView5, ImageView imageView6, ImageView imageView7, ImageView imageView8, ImageView imageView9, ImageView imageView10, LinearLayout linearLayout, ConstraintLayout constraintLayout2, ConstraintLayout constraintLayout3, LinearLayout linearLayout2, ConstraintLayout constraintLayout4, OverlayView overlayView, PlayerView playerView, SeekBar seekBar, ImageView imageView11, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, View view2) {
        this.rootView = constraintLayout;
        this.bgSelectVideo = view;
        this.cachingSpinner = progressBar;
        this.groupEmpty = group;
        this.groupPlaying = group2;
        this.imgBluetooth = imageView;
        this.imgPanelGesture = imageView2;
        this.imgSelectVideo = imageView3;
        this.imgVideoEmpty = imageView4;
        this.imgVideoLoop = imageView5;
        this.imgVideoNext = imageView6;
        this.imgVideoPanel = imageView7;
        this.imgVideoPlay = imageView8;
        this.imgVideoPrevious = imageView9;
        this.imgVideoSelect = imageView10;
        this.layoutCachingHint = linearLayout;
        this.layoutControl = constraintLayout2;
        this.layoutSpeed = constraintLayout3;
        this.layoutTitle = linearLayout2;
        this.main = constraintLayout4;
        this.overlay = overlayView;
        this.playView = playerView;
        this.seekVideo = seekBar;
        this.titleImgBack = imageView11;
        this.titleTxtName = textView;
        this.txtCachingHint = textView2;
        this.txtSelectVideo = textView3;
        this.txtSpeed = textView4;
        this.txtVideoEmpty = textView5;
        this.txtVideoTimeCurrent = textView6;
        this.txtVideoTimeTotal = textView7;
        this.viewHeight = view2;
    }

    public static ActivityPlayVideoBinding bind(View view) {
        View findChildViewById;
        int i = R.id.bg_select_video;
        View findChildViewById2 = ViewBindings.findChildViewById(view, i);
        if (findChildViewById2 != null) {
            i = R.id.caching_spinner;
            ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
            if (progressBar != null) {
                i = R.id.group_empty;
                Group group = (Group) ViewBindings.findChildViewById(view, i);
                if (group != null) {
                    i = R.id.group_playing;
                    Group group2 = (Group) ViewBindings.findChildViewById(view, i);
                    if (group2 != null) {
                        i = R.id.img_bluetooth;
                        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView != null) {
                            i = R.id.img_panel_gesture;
                            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView2 != null) {
                                i = R.id.imgSelectVideo;
                                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView3 != null) {
                                    i = R.id.img_video_empty;
                                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView4 != null) {
                                        i = R.id.img_video_loop;
                                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView5 != null) {
                                            i = R.id.img_video_next;
                                            ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                            if (imageView6 != null) {
                                                i = R.id.img_video_panel;
                                                ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                if (imageView7 != null) {
                                                    i = R.id.img_video_play;
                                                    ImageView imageView8 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                    if (imageView8 != null) {
                                                        i = R.id.img_video_previous;
                                                        ImageView imageView9 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                        if (imageView9 != null) {
                                                            i = R.id.img_video_select;
                                                            ImageView imageView10 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                            if (imageView10 != null) {
                                                                i = R.id.layout_caching_hint;
                                                                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                if (linearLayout != null) {
                                                                    i = R.id.layout_control;
                                                                    ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                    if (constraintLayout != null) {
                                                                        i = R.id.layout_speed;
                                                                        ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                                                        if (constraintLayout2 != null) {
                                                                            i = R.id.layout_title;
                                                                            LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                            if (linearLayout2 != null) {
                                                                                ConstraintLayout constraintLayout3 = (ConstraintLayout) view;
                                                                                i = R.id.overlay;
                                                                                OverlayView overlayView = (OverlayView) ViewBindings.findChildViewById(view, i);
                                                                                if (overlayView != null) {
                                                                                    i = R.id.playView;
                                                                                    PlayerView playerView = (PlayerView) ViewBindings.findChildViewById(view, i);
                                                                                    if (playerView != null) {
                                                                                        i = R.id.seek_video;
                                                                                        SeekBar seekBar = (SeekBar) ViewBindings.findChildViewById(view, i);
                                                                                        if (seekBar != null) {
                                                                                            i = R.id.title_img_back;
                                                                                            ImageView imageView11 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                            if (imageView11 != null) {
                                                                                                i = R.id.title_txt_name;
                                                                                                TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView != null) {
                                                                                                    i = R.id.txt_caching_hint;
                                                                                                    TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                    if (textView2 != null) {
                                                                                                        i = R.id.txtSelectVideo;
                                                                                                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (textView3 != null) {
                                                                                                            i = R.id.txt_speed;
                                                                                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                            if (textView4 != null) {
                                                                                                                i = R.id.txt_video_empty;
                                                                                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                if (textView5 != null) {
                                                                                                                    i = R.id.txt_video_time_current;
                                                                                                                    TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                    if (textView6 != null) {
                                                                                                                        i = R.id.txt_video_time_total;
                                                                                                                        TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                        if (textView7 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.view_height))) != null) {
                                                                                                                            return new ActivityPlayVideoBinding(constraintLayout3, findChildViewById2, progressBar, group, group2, imageView, imageView2, imageView3, imageView4, imageView5, imageView6, imageView7, imageView8, imageView9, imageView10, linearLayout, constraintLayout, constraintLayout2, linearLayout2, constraintLayout3, overlayView, playerView, seekBar, imageView11, textView, textView2, textView3, textView4, textView5, textView6, textView7, findChildViewById);
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

    public static ActivityPlayVideoBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityPlayVideoBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_play_video, viewGroup, false);
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
