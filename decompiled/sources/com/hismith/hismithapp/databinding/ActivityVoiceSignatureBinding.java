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
import com.hismith.hismithapp.view.GradientRoundProgress;

/* loaded from: classes3.dex */
public final class ActivityVoiceSignatureBinding implements ViewBinding {
    public final ImageView imgRecord;
    public final ImageView imgRepeal;
    public final ImageView imgSave;
    public final LinearLayout layoutDelete;
    public final LinearLayout layoutSave;
    private final LinearLayout rootView;
    public final TextView titleTxtName;
    public final ImageView voiceImgPlay;
    public final ImageView voiceImgWave;
    public final GradientRoundProgress voiceProgressSeconds;
    public final TextView voiceTxtSecondes;

    private ActivityVoiceSignatureBinding(LinearLayout linearLayout, ImageView imageView, ImageView imageView2, ImageView imageView3, LinearLayout linearLayout2, LinearLayout linearLayout3, TextView textView, ImageView imageView4, ImageView imageView5, GradientRoundProgress gradientRoundProgress, TextView textView2) {
        this.rootView = linearLayout;
        this.imgRecord = imageView;
        this.imgRepeal = imageView2;
        this.imgSave = imageView3;
        this.layoutDelete = linearLayout2;
        this.layoutSave = linearLayout3;
        this.titleTxtName = textView;
        this.voiceImgPlay = imageView4;
        this.voiceImgWave = imageView5;
        this.voiceProgressSeconds = gradientRoundProgress;
        this.voiceTxtSecondes = textView2;
    }

    public static ActivityVoiceSignatureBinding bind(View view) {
        int i = R.id.img_record;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.img_repeal;
            ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView2 != null) {
                i = R.id.img_save;
                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView3 != null) {
                    i = R.id.layout_delete;
                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout != null) {
                        i = R.id.layout_save;
                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                        if (linearLayout2 != null) {
                            i = R.id.title_txt_name;
                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView != null) {
                                i = R.id.voice_img_play;
                                ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView4 != null) {
                                    i = R.id.voice_img_wave;
                                    ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView5 != null) {
                                        i = R.id.voice_progress_seconds;
                                        GradientRoundProgress gradientRoundProgress = (GradientRoundProgress) ViewBindings.findChildViewById(view, i);
                                        if (gradientRoundProgress != null) {
                                            i = R.id.voice_txt_secondes;
                                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView2 != null) {
                                                return new ActivityVoiceSignatureBinding((LinearLayout) view, imageView, imageView2, imageView3, linearLayout, linearLayout2, textView, imageView4, imageView5, gradientRoundProgress, textView2);
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

    public static ActivityVoiceSignatureBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityVoiceSignatureBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_voice_signature, viewGroup, false);
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
