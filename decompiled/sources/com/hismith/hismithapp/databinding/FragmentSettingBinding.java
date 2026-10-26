package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class FragmentSettingBinding implements ViewBinding {
    private final LinearLayout rootView;
    public final ImageView settingImgAddFriendOn;
    public final ImageView settingImgPlayLogs;
    public final ImageView settingImgStealth;
    public final LinearLayout settingLayoutAboutus;
    public final LinearLayout settingLayoutChangeLanguage;
    public final LinearLayout settingLayoutClearCache;
    public final LinearLayout settingLayoutContactus;
    public final LinearLayout settingLayoutLogout;
    public final LinearLayout settingLayoutMessageRemind;
    public final LinearLayout settingLayoutSecurity;
    public final LinearLayout settingLayoutViewSex;
    public final View settingResetLine;
    public final TextView settingTxtAboutus;
    public final TextView settingTxtAddFriendOn;
    public final TextView settingTxtCacheValue;
    public final TextView settingTxtChangeLanguage;
    public final TextView settingTxtClearCache;
    public final TextView settingTxtContactus;
    public final TextView settingTxtLogout;
    public final TextView settingTxtMessageRemind;
    public final TextView settingTxtPlayLogs;
    public final TextView settingTxtSecurity;
    public final TextView settingTxtStealth;
    public final TextView settingTxtViewSex;

    private FragmentSettingBinding(LinearLayout linearLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, LinearLayout linearLayout2, LinearLayout linearLayout3, LinearLayout linearLayout4, LinearLayout linearLayout5, LinearLayout linearLayout6, LinearLayout linearLayout7, LinearLayout linearLayout8, LinearLayout linearLayout9, View view, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9, TextView textView10, TextView textView11, TextView textView12) {
        this.rootView = linearLayout;
        this.settingImgAddFriendOn = imageView;
        this.settingImgPlayLogs = imageView2;
        this.settingImgStealth = imageView3;
        this.settingLayoutAboutus = linearLayout2;
        this.settingLayoutChangeLanguage = linearLayout3;
        this.settingLayoutClearCache = linearLayout4;
        this.settingLayoutContactus = linearLayout5;
        this.settingLayoutLogout = linearLayout6;
        this.settingLayoutMessageRemind = linearLayout7;
        this.settingLayoutSecurity = linearLayout8;
        this.settingLayoutViewSex = linearLayout9;
        this.settingResetLine = view;
        this.settingTxtAboutus = textView;
        this.settingTxtAddFriendOn = textView2;
        this.settingTxtCacheValue = textView3;
        this.settingTxtChangeLanguage = textView4;
        this.settingTxtClearCache = textView5;
        this.settingTxtContactus = textView6;
        this.settingTxtLogout = textView7;
        this.settingTxtMessageRemind = textView8;
        this.settingTxtPlayLogs = textView9;
        this.settingTxtSecurity = textView10;
        this.settingTxtStealth = textView11;
        this.settingTxtViewSex = textView12;
    }

    public static FragmentSettingBinding bind(View view) {
        View findChildViewById;
        int i = R.id.setting_img_addFriendOn;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.setting_img_playLogs;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.setting_img_stealth;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.setting_layout_aboutus;
                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout != null) {
                        i = R.id.setting_layout_changeLanguage;
                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                        if (linearLayout2 != null) {
                            i = R.id.setting_layout_clearCache;
                            LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                            if (linearLayout3 != null) {
                                i = R.id.setting_layout_contactus;
                                LinearLayout linearLayout4 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout4 != null) {
                                    i = R.id.setting_layout_logout;
                                    LinearLayout linearLayout5 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                    if (linearLayout5 != null) {
                                        i = R.id.setting_layout_messageRemind;
                                        LinearLayout linearLayout6 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                        if (linearLayout6 != null) {
                                            i = R.id.setting_layout_security;
                                            LinearLayout linearLayout7 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                            if (linearLayout7 != null) {
                                                i = R.id.setting_layout_viewSex;
                                                LinearLayout linearLayout8 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                if (linearLayout8 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.setting_reset_line))) != null) {
                                                    i = R.id.setting_txt_aboutus;
                                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView != null) {
                                                        i = R.id.setting_txt_addFriendOn;
                                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView2 != null) {
                                                            i = R.id.setting_txt_cacheValue;
                                                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView3 != null) {
                                                                i = R.id.setting_txt_changeLanguage;
                                                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView4 != null) {
                                                                    i = R.id.setting_txt_clearCache;
                                                                    TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView5 != null) {
                                                                        i = R.id.setting_txt_contactus;
                                                                        TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView6 != null) {
                                                                            i = R.id.setting_txt_logout;
                                                                            TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView7 != null) {
                                                                                i = R.id.setting_txt_messageRemind;
                                                                                TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView8 != null) {
                                                                                    i = R.id.setting_txt_playLogs;
                                                                                    TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                    if (textView9 != null) {
                                                                                        i = R.id.setting_txt_security;
                                                                                        TextView textView10 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView10 != null) {
                                                                                            i = R.id.setting_txt_stealth;
                                                                                            TextView textView11 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (textView11 != null) {
                                                                                                i = R.id.setting_txt_viewSex;
                                                                                                TextView textView12 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView12 != null) {
                                                                                                    return new FragmentSettingBinding((LinearLayout) view, imageView, imageView2, imageView3, linearLayout, linearLayout2, linearLayout3, linearLayout4, linearLayout5, linearLayout6, linearLayout7, linearLayout8, findChildViewById, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8, textView9, textView10, textView11, textView12);
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

    public static FragmentSettingBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static FragmentSettingBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.fragment_setting, viewGroup, false);
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
