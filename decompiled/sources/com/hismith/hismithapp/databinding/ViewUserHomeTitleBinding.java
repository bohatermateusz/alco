package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import androidx.viewpager.widget.ViewPager;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;

/* loaded from: classes3.dex */
public final class ViewUserHomeTitleBinding implements ViewBinding {
    public final ImageView detailsImgGender;
    public final CircleImageView detailsImgHead;
    public final ImageView detailsImgLogo;
    public final ImageView detailsImgVoice;
    public final ImageView detailsImgVoiceB;
    public final LinearLayout detailsLayoutChat;
    public final LinearLayout detailsLayoutDots;
    public final LinearLayout detailsLayoutEdit;
    public final LinearLayout detailsLayoutFollow;
    public final LinearLayout detailsLayoutInterestedIn;
    public final RelativeLayout detailsLayoutPhoto;
    public final LinearLayout detailsLayoutVoice;
    public final ProgressBar detailsProgressFollow;
    public final TextView detailsTxtBios;
    public final TextView detailsTxtChat;
    public final TextView detailsTxtConstellation;
    public final TextView detailsTxtEdit;
    public final TextView detailsTxtFans;
    public final TextView detailsTxtFansLabel;
    public final TextView detailsTxtFocus;
    public final TextView detailsTxtFocusLabel;
    public final TextView detailsTxtFollow;
    public final TextView detailsTxtId;
    public final TextView detailsTxtIdLabel;
    public final TextView detailsTxtInterestedIn;
    public final TextView detailsTxtInterestedInLabel;
    public final TextView detailsTxtLocation;
    public final TextView detailsTxtLocationLabel;
    public final TextView detailsTxtName;
    public final TextView detailsTxtVoiceTime;
    public final ViewPager detailsViewPager;
    private final LinearLayout rootView;

    private ViewUserHomeTitleBinding(LinearLayout linearLayout, ImageView imageView, CircleImageView circleImageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, LinearLayout linearLayout2, LinearLayout linearLayout3, LinearLayout linearLayout4, LinearLayout linearLayout5, LinearLayout linearLayout6, RelativeLayout relativeLayout, LinearLayout linearLayout7, ProgressBar progressBar, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9, TextView textView10, TextView textView11, TextView textView12, TextView textView13, TextView textView14, TextView textView15, TextView textView16, TextView textView17, ViewPager viewPager) {
        this.rootView = linearLayout;
        this.detailsImgGender = imageView;
        this.detailsImgHead = circleImageView;
        this.detailsImgLogo = imageView2;
        this.detailsImgVoice = imageView3;
        this.detailsImgVoiceB = imageView4;
        this.detailsLayoutChat = linearLayout2;
        this.detailsLayoutDots = linearLayout3;
        this.detailsLayoutEdit = linearLayout4;
        this.detailsLayoutFollow = linearLayout5;
        this.detailsLayoutInterestedIn = linearLayout6;
        this.detailsLayoutPhoto = relativeLayout;
        this.detailsLayoutVoice = linearLayout7;
        this.detailsProgressFollow = progressBar;
        this.detailsTxtBios = textView;
        this.detailsTxtChat = textView2;
        this.detailsTxtConstellation = textView3;
        this.detailsTxtEdit = textView4;
        this.detailsTxtFans = textView5;
        this.detailsTxtFansLabel = textView6;
        this.detailsTxtFocus = textView7;
        this.detailsTxtFocusLabel = textView8;
        this.detailsTxtFollow = textView9;
        this.detailsTxtId = textView10;
        this.detailsTxtIdLabel = textView11;
        this.detailsTxtInterestedIn = textView12;
        this.detailsTxtInterestedInLabel = textView13;
        this.detailsTxtLocation = textView14;
        this.detailsTxtLocationLabel = textView15;
        this.detailsTxtName = textView16;
        this.detailsTxtVoiceTime = textView17;
        this.detailsViewPager = viewPager;
    }

    public static ViewUserHomeTitleBinding bind(View view) {
        int i = R.id.details_img_gender;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.details_img_head;
            CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
            if (circleImageView != null) {
                i = R.id.details_img_logo;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.details_img_voice;
                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                    if (imageView3 != null) {
                        i = R.id.details_img_voice_b;
                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                        if (imageView4 != null) {
                            i = R.id.details_layout_chat;
                            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                            if (linearLayout != null) {
                                i = R.id.details_layout_dots;
                                LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout2 != null) {
                                    i = R.id.details_layout_edit;
                                    LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                    if (linearLayout3 != null) {
                                        i = R.id.details_layout_follow;
                                        LinearLayout linearLayout4 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                        if (linearLayout4 != null) {
                                            i = R.id.details_layout_interestedIn;
                                            LinearLayout linearLayout5 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                            if (linearLayout5 != null) {
                                                i = R.id.details_layout_photo;
                                                RelativeLayout relativeLayout = (RelativeLayout) ViewBindings.findChildViewById(view, i);
                                                if (relativeLayout != null) {
                                                    i = R.id.details_layout_voice;
                                                    LinearLayout linearLayout6 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                    if (linearLayout6 != null) {
                                                        i = R.id.details_progress_follow;
                                                        ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                        if (progressBar != null) {
                                                            i = R.id.details_txt_bios;
                                                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView != null) {
                                                                i = R.id.details_txt_chat;
                                                                TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView2 != null) {
                                                                    i = R.id.details_txt_constellation;
                                                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView3 != null) {
                                                                        i = R.id.details_txt_edit;
                                                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView4 != null) {
                                                                            i = R.id.details_txt_fans;
                                                                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView5 != null) {
                                                                                i = R.id.details_txt_fans_label;
                                                                                TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView6 != null) {
                                                                                    i = R.id.details_txt_focus;
                                                                                    TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                    if (textView7 != null) {
                                                                                        i = R.id.details_txt_focus_label;
                                                                                        TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView8 != null) {
                                                                                            i = R.id.details_txt_follow;
                                                                                            TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (textView9 != null) {
                                                                                                i = R.id.details_txt_id;
                                                                                                TextView textView10 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView10 != null) {
                                                                                                    i = R.id.details_txt_id_label;
                                                                                                    TextView textView11 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                    if (textView11 != null) {
                                                                                                        i = R.id.details_txt_interestedIn;
                                                                                                        TextView textView12 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (textView12 != null) {
                                                                                                            i = R.id.details_txt_interestedIn_label;
                                                                                                            TextView textView13 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                            if (textView13 != null) {
                                                                                                                i = R.id.details_txt_location;
                                                                                                                TextView textView14 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                if (textView14 != null) {
                                                                                                                    i = R.id.details_txt_location_label;
                                                                                                                    TextView textView15 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                    if (textView15 != null) {
                                                                                                                        i = R.id.details_txt_name;
                                                                                                                        TextView textView16 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                        if (textView16 != null) {
                                                                                                                            i = R.id.details_txt_voice_time;
                                                                                                                            TextView textView17 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                            if (textView17 != null) {
                                                                                                                                i = R.id.details_view_pager;
                                                                                                                                ViewPager viewPager = (ViewPager) ViewBindings.findChildViewById(view, i);
                                                                                                                                if (viewPager != null) {
                                                                                                                                    return new ViewUserHomeTitleBinding((LinearLayout) view, imageView, circleImageView, imageView2, imageView3, imageView4, linearLayout, linearLayout2, linearLayout3, linearLayout4, linearLayout5, relativeLayout, linearLayout6, progressBar, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8, textView9, textView10, textView11, textView12, textView13, textView14, textView15, textView16, textView17, viewPager);
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

    public static ViewUserHomeTitleBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ViewUserHomeTitleBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.view_user_home_title, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public LinearLayout getRoot() {
        return this.rootView;
    }
}
