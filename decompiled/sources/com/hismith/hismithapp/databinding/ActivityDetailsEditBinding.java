package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.GridLayout;

/* loaded from: classes3.dex */
public final class ActivityDetailsEditBinding implements ViewBinding {
    public final TextView editEditBio;
    public final TextView editEditBirthday;
    public final TextView editEditEmail;
    public final EditText editEditName;
    public final EditText editEditPhone;
    public final GridLayout editGridLayout;
    public final LinearLayout editLayoutBio;
    public final LinearLayout editLayoutHasVoiceSignature;
    public final FrameLayout editLayoutVoiceSignature;
    public final TextView editTxtBio;
    public final TextView editTxtBirthday;
    public final TextView editTxtEmail;
    public final TextView editTxtFrom;
    public final TextView editTxtFromLabel;
    public final TextView editTxtGender;
    public final TextView editTxtInterested;
    public final TextView editTxtName;
    public final TextView editTxtPhone;
    public final TextView editTxtSave;
    public final TextView editTxtVoiceLength;
    public final TextView editTxtVoiceSignature;
    public final TextView editTxtVoiceSignatureLabel;
    public final LinearLayout layoutRoot;
    private final LinearLayout rootView;
    public final TextView titleTxtName;
    public final TextView txtGenderFemale;
    public final TextView txtGenderMale;
    public final TextView txtGenderTrans;
    public final TextView txtInterestedFemale;
    public final TextView txtInterestedMale;
    public final TextView txtInterestedTrans;

    private ActivityDetailsEditBinding(LinearLayout linearLayout, TextView textView, TextView textView2, TextView textView3, EditText editText, EditText editText2, GridLayout gridLayout, LinearLayout linearLayout2, LinearLayout linearLayout3, FrameLayout frameLayout, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9, TextView textView10, TextView textView11, TextView textView12, TextView textView13, TextView textView14, TextView textView15, TextView textView16, LinearLayout linearLayout4, TextView textView17, TextView textView18, TextView textView19, TextView textView20, TextView textView21, TextView textView22, TextView textView23) {
        this.rootView = linearLayout;
        this.editEditBio = textView;
        this.editEditBirthday = textView2;
        this.editEditEmail = textView3;
        this.editEditName = editText;
        this.editEditPhone = editText2;
        this.editGridLayout = gridLayout;
        this.editLayoutBio = linearLayout2;
        this.editLayoutHasVoiceSignature = linearLayout3;
        this.editLayoutVoiceSignature = frameLayout;
        this.editTxtBio = textView4;
        this.editTxtBirthday = textView5;
        this.editTxtEmail = textView6;
        this.editTxtFrom = textView7;
        this.editTxtFromLabel = textView8;
        this.editTxtGender = textView9;
        this.editTxtInterested = textView10;
        this.editTxtName = textView11;
        this.editTxtPhone = textView12;
        this.editTxtSave = textView13;
        this.editTxtVoiceLength = textView14;
        this.editTxtVoiceSignature = textView15;
        this.editTxtVoiceSignatureLabel = textView16;
        this.layoutRoot = linearLayout4;
        this.titleTxtName = textView17;
        this.txtGenderFemale = textView18;
        this.txtGenderMale = textView19;
        this.txtGenderTrans = textView20;
        this.txtInterestedFemale = textView21;
        this.txtInterestedMale = textView22;
        this.txtInterestedTrans = textView23;
    }

    public static ActivityDetailsEditBinding bind(View view) {
        int i = R.id.edit_edit_bio;
        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
        if (textView != null) {
            i = R.id.edit_edit_birthday;
            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
            if (textView2 != null) {
                i = R.id.edit_edit_email;
                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                if (textView3 != null) {
                    i = R.id.edit_edit_name;
                    EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                    if (editText != null) {
                        i = R.id.edit_edit_phone;
                        EditText editText2 = (EditText) ViewBindings.findChildViewById(view, i);
                        if (editText2 != null) {
                            i = R.id.edit_grid_layout;
                            GridLayout gridLayout = (GridLayout) ViewBindings.findChildViewById(view, i);
                            if (gridLayout != null) {
                                i = R.id.edit_layout_bio;
                                LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                if (linearLayout != null) {
                                    i = R.id.edit_layout_has_voiceSignature;
                                    LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                    if (linearLayout2 != null) {
                                        i = R.id.edit_layout_voiceSignature;
                                        FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                        if (frameLayout != null) {
                                            i = R.id.edit_txt_bio;
                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView4 != null) {
                                                i = R.id.edit_txt_birthday;
                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView5 != null) {
                                                    i = R.id.edit_txt_email;
                                                    TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView6 != null) {
                                                        i = R.id.edit_txt_from;
                                                        TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView7 != null) {
                                                            i = R.id.edit_txt_from_label;
                                                            TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView8 != null) {
                                                                i = R.id.edit_txt_gender;
                                                                TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                if (textView9 != null) {
                                                                    i = R.id.edit_txt_interested;
                                                                    TextView textView10 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                    if (textView10 != null) {
                                                                        i = R.id.edit_txt_name;
                                                                        TextView textView11 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                        if (textView11 != null) {
                                                                            i = R.id.edit_txt_phone;
                                                                            TextView textView12 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                            if (textView12 != null) {
                                                                                i = R.id.edit_txt_save;
                                                                                TextView textView13 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                if (textView13 != null) {
                                                                                    i = R.id.edit_txt_voice_length;
                                                                                    TextView textView14 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                    if (textView14 != null) {
                                                                                        i = R.id.edit_txt_voiceSignature;
                                                                                        TextView textView15 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                        if (textView15 != null) {
                                                                                            i = R.id.edit_txt_voiceSignature_label;
                                                                                            TextView textView16 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                            if (textView16 != null) {
                                                                                                LinearLayout linearLayout3 = (LinearLayout) view;
                                                                                                i = R.id.title_txt_name;
                                                                                                TextView textView17 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                if (textView17 != null) {
                                                                                                    i = R.id.txt_gender_female;
                                                                                                    TextView textView18 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                    if (textView18 != null) {
                                                                                                        i = R.id.txt_gender_male;
                                                                                                        TextView textView19 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                        if (textView19 != null) {
                                                                                                            i = R.id.txt_gender_trans;
                                                                                                            TextView textView20 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                            if (textView20 != null) {
                                                                                                                i = R.id.txt_interested_female;
                                                                                                                TextView textView21 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                if (textView21 != null) {
                                                                                                                    i = R.id.txt_interested_male;
                                                                                                                    TextView textView22 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                    if (textView22 != null) {
                                                                                                                        i = R.id.txt_interested_trans;
                                                                                                                        TextView textView23 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                        if (textView23 != null) {
                                                                                                                            return new ActivityDetailsEditBinding(linearLayout3, textView, textView2, textView3, editText, editText2, gridLayout, linearLayout, linearLayout2, frameLayout, textView4, textView5, textView6, textView7, textView8, textView9, textView10, textView11, textView12, textView13, textView14, textView15, textView16, linearLayout3, textView17, textView18, textView19, textView20, textView21, textView22, textView23);
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

    public static ActivityDetailsEditBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityDetailsEditBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_details_edit, viewGroup, false);
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
