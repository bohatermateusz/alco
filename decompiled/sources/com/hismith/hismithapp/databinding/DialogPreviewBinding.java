package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.LineChartPlayView3;

/* loaded from: classes3.dex */
public final class DialogPreviewBinding implements ViewBinding {
    public final ImageView dialogPreviewImgClose;
    public final FrameLayout dialogPreviewLayoutTime;
    public final LineChartPlayView3 dialogPreviewLineChartPlay;
    public final SeekBar dialogPreviewSeekBar;
    public final TextView dialogPreviewTxtEndTime;
    public final TextView dialogPreviewTxtStartTime;
    public final LinearLayout layoutMark;
    private final FrameLayout rootView;
    public final TextView txtModeDepth;
    public final TextView txtModeMaxStroke;
    public final TextView txtModeSmooth;
    public final TextView txtModeSpeed;

    private DialogPreviewBinding(FrameLayout frameLayout, ImageView imageView, FrameLayout frameLayout2, LineChartPlayView3 lineChartPlayView3, SeekBar seekBar, TextView textView, TextView textView2, LinearLayout linearLayout, TextView textView3, TextView textView4, TextView textView5, TextView textView6) {
        this.rootView = frameLayout;
        this.dialogPreviewImgClose = imageView;
        this.dialogPreviewLayoutTime = frameLayout2;
        this.dialogPreviewLineChartPlay = lineChartPlayView3;
        this.dialogPreviewSeekBar = seekBar;
        this.dialogPreviewTxtEndTime = textView;
        this.dialogPreviewTxtStartTime = textView2;
        this.layoutMark = linearLayout;
        this.txtModeDepth = textView3;
        this.txtModeMaxStroke = textView4;
        this.txtModeSmooth = textView5;
        this.txtModeSpeed = textView6;
    }

    public static DialogPreviewBinding bind(View view) {
        int i = R.id.dialog_preview_img_close;
        ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
        if (imageView != null) {
            i = R.id.dialog_preview_layout_time;
            FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
            if (frameLayout != null) {
                i = R.id.dialog_preview_lineChartPlay;
                LineChartPlayView3 lineChartPlayView3 = (LineChartPlayView3) ViewBindings.findChildViewById(view, i);
                if (lineChartPlayView3 != null) {
                    i = R.id.dialog_preview_seek_bar;
                    SeekBar seekBar = (SeekBar) ViewBindings.findChildViewById(view, i);
                    if (seekBar != null) {
                        i = R.id.dialog_preview_txt_end_time;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            i = R.id.dialog_preview_txt_start_time;
                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView2 != null) {
                                i = R.id.layout_mark;
                                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout != null) {
                                    i = R.id.txt_mode_depth;
                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView3 != null) {
                                        i = R.id.txt_mode_max_stroke;
                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView4 != null) {
                                            i = R.id.txt_mode_smooth;
                                            TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView5 != null) {
                                                i = R.id.txt_mode_speed;
                                                TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView6 != null) {
                                                    return new DialogPreviewBinding((FrameLayout) view, imageView, frameLayout, lineChartPlayView3, seekBar, textView, textView2, linearLayout, textView3, textView4, textView5, textView6);
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

    public static DialogPreviewBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogPreviewBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_preview, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public FrameLayout getRoot() {
        return this.rootView;
    }
}
