package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatTextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.recyclerview.widget.RecyclerView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.view.CircleImageView;
import com.hismith.hismithapp.view.CustomSeekbar;
import com.hismith.hismithapp.view.CustomWebView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;

/* loaded from: classes3.dex */
public final class ActivityLineRoom2Binding implements ViewBinding {
    public final Button btnMessageSwitch;
    public final ImageView imgPosition;
    public final ImageView imgRunMode;
    public final FrameLayout layoutPositionMax;
    public final FrameLayout layoutPositionMin;
    public final ConstraintLayout layoutSmsTouchPosition;
    public final FrameLayout layoutSpeedMax;
    public final ConstraintLayout layoutTouchPosition;
    public final EditText lineRoomEditMessage;
    public final FrameLayout lineRoomFragmentPreview;
    public final ImageView lineRoomImgChat;
    public final ImageView lineRoomImgExit;
    public final ImageView lineRoomImgMode;
    public final CircleImageView lineRoomImgOwnerHead;
    public final ImageView lineRoomImgRoomPublic;
    public final ImageView lineRoomImgSound;
    public final ImageView lineRoomImgStart;
    public final ImageView lineRoomImgUserCrown;
    public final ImageView lineRoomImgUserInvite;
    public final ImageView lineRoomImgVoice;
    public final ImageView lineRoomLayoutBg;
    public final FrameLayout lineRoomLayoutMaxSpeed;
    public final FrameLayout lineRoomLayoutMaxVibrate;
    public final LinearLayout lineRoomLayoutMessage;
    public final LinearLayout lineRoomLayoutSmsInfo;
    public final RelativeLayout lineRoomLayoutSpeedInfo;
    public final LinearLayout lineRoomLayoutUserCount;
    public final RelativeLayout lineRoomLayoutUserInfo;
    public final RelativeLayout lineRoomLayoutVibrateInfo;
    public final View lineRoomLine;
    public final PlayView_Four lineRoomPlayView;
    public final PlayView_Four lineRoomPlayViewVibrate;
    public final VerticalProgressBar lineRoomProgressSpeed;
    public final VerticalProgressBar lineRoomProgressVibrate;
    public final RecyclerView lineRoomRecyclerMessage;
    public final RecyclerView lineRoomRecyclerUser;
    public final TextView lineRoomTxtMaxSpeed;
    public final TextView lineRoomTxtMaxVibrate;
    public final TextView lineRoomTxtMinSpeed;
    public final TextView lineRoomTxtMinVibrate;
    public final TextView lineRoomTxtOwnerName;
    public final TextView lineRoomTxtRoomId;
    public final TextView lineRoomTxtSpeed;
    public final TextView lineRoomTxtUserCount;
    public final TextView lineRoomTxtVibrate;
    public final ProgressBar progressPosition;
    public final ProgressBar progressSpeed;
    private final LinearLayout rootView;
    public final CustomSeekbar seekSmooth;
    public final TextView txtPosition;
    public final TextView txtPositionLabel;
    public final AppCompatTextView txtPositionMax;
    public final AppCompatTextView txtPositionMin;
    public final TextView txtSmooth;
    public final TextView txtSmoothLabel;
    public final TextView txtSpeed;
    public final TextView txtSpeedLabel;
    public final AppCompatTextView txtSpeedMax;
    public final ViewNetworkErrorBinding viewNetworkError;
    public final CustomWebView webView;
    public final CustomWebView webViewRocket;

    private ActivityLineRoom2Binding(LinearLayout linearLayout, Button button, ImageView imageView, ImageView imageView2, FrameLayout frameLayout, FrameLayout frameLayout2, ConstraintLayout constraintLayout, FrameLayout frameLayout3, ConstraintLayout constraintLayout2, EditText editText, FrameLayout frameLayout4, ImageView imageView3, ImageView imageView4, ImageView imageView5, CircleImageView circleImageView, ImageView imageView6, ImageView imageView7, ImageView imageView8, ImageView imageView9, ImageView imageView10, ImageView imageView11, ImageView imageView12, FrameLayout frameLayout5, FrameLayout frameLayout6, LinearLayout linearLayout2, LinearLayout linearLayout3, RelativeLayout relativeLayout, LinearLayout linearLayout4, RelativeLayout relativeLayout2, RelativeLayout relativeLayout3, View view, PlayView_Four playView_Four, PlayView_Four playView_Four2, VerticalProgressBar verticalProgressBar, VerticalProgressBar verticalProgressBar2, RecyclerView recyclerView, RecyclerView recyclerView2, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9, ProgressBar progressBar, ProgressBar progressBar2, CustomSeekbar customSeekbar, TextView textView10, TextView textView11, AppCompatTextView appCompatTextView, AppCompatTextView appCompatTextView2, TextView textView12, TextView textView13, TextView textView14, TextView textView15, AppCompatTextView appCompatTextView3, ViewNetworkErrorBinding viewNetworkErrorBinding, CustomWebView customWebView, CustomWebView customWebView2) {
        this.rootView = linearLayout;
        this.btnMessageSwitch = button;
        this.imgPosition = imageView;
        this.imgRunMode = imageView2;
        this.layoutPositionMax = frameLayout;
        this.layoutPositionMin = frameLayout2;
        this.layoutSmsTouchPosition = constraintLayout;
        this.layoutSpeedMax = frameLayout3;
        this.layoutTouchPosition = constraintLayout2;
        this.lineRoomEditMessage = editText;
        this.lineRoomFragmentPreview = frameLayout4;
        this.lineRoomImgChat = imageView3;
        this.lineRoomImgExit = imageView4;
        this.lineRoomImgMode = imageView5;
        this.lineRoomImgOwnerHead = circleImageView;
        this.lineRoomImgRoomPublic = imageView6;
        this.lineRoomImgSound = imageView7;
        this.lineRoomImgStart = imageView8;
        this.lineRoomImgUserCrown = imageView9;
        this.lineRoomImgUserInvite = imageView10;
        this.lineRoomImgVoice = imageView11;
        this.lineRoomLayoutBg = imageView12;
        this.lineRoomLayoutMaxSpeed = frameLayout5;
        this.lineRoomLayoutMaxVibrate = frameLayout6;
        this.lineRoomLayoutMessage = linearLayout2;
        this.lineRoomLayoutSmsInfo = linearLayout3;
        this.lineRoomLayoutSpeedInfo = relativeLayout;
        this.lineRoomLayoutUserCount = linearLayout4;
        this.lineRoomLayoutUserInfo = relativeLayout2;
        this.lineRoomLayoutVibrateInfo = relativeLayout3;
        this.lineRoomLine = view;
        this.lineRoomPlayView = playView_Four;
        this.lineRoomPlayViewVibrate = playView_Four2;
        this.lineRoomProgressSpeed = verticalProgressBar;
        this.lineRoomProgressVibrate = verticalProgressBar2;
        this.lineRoomRecyclerMessage = recyclerView;
        this.lineRoomRecyclerUser = recyclerView2;
        this.lineRoomTxtMaxSpeed = textView;
        this.lineRoomTxtMaxVibrate = textView2;
        this.lineRoomTxtMinSpeed = textView3;
        this.lineRoomTxtMinVibrate = textView4;
        this.lineRoomTxtOwnerName = textView5;
        this.lineRoomTxtRoomId = textView6;
        this.lineRoomTxtSpeed = textView7;
        this.lineRoomTxtUserCount = textView8;
        this.lineRoomTxtVibrate = textView9;
        this.progressPosition = progressBar;
        this.progressSpeed = progressBar2;
        this.seekSmooth = customSeekbar;
        this.txtPosition = textView10;
        this.txtPositionLabel = textView11;
        this.txtPositionMax = appCompatTextView;
        this.txtPositionMin = appCompatTextView2;
        this.txtSmooth = textView12;
        this.txtSmoothLabel = textView13;
        this.txtSpeed = textView14;
        this.txtSpeedLabel = textView15;
        this.txtSpeedMax = appCompatTextView3;
        this.viewNetworkError = viewNetworkErrorBinding;
        this.webView = customWebView;
        this.webViewRocket = customWebView2;
    }

    public static ActivityLineRoom2Binding bind(View view) {
        View findChildViewById;
        View findChildViewById2;
        int i = R.id.btn_message_switch;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.img_position;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.img_run_mode;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.layout_position_max;
                    FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                    if (frameLayout != null) {
                        i = R.id.layout_position_min;
                        FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                        if (frameLayout2 != null) {
                            i = R.id.layout_sms_touch_position;
                            ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout != null) {
                                i = R.id.layout_speed_max;
                                FrameLayout frameLayout3 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                if (frameLayout3 != null) {
                                    i = R.id.layout_touch_position;
                                    ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                    if (constraintLayout2 != null) {
                                        i = R.id.line_room_edit_message;
                                        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                                        if (editText != null) {
                                            i = R.id.line_room_fragment_preview;
                                            FrameLayout frameLayout4 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                            if (frameLayout4 != null) {
                                                i = R.id.line_room_img_chat;
                                                ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                if (imageView3 != null) {
                                                    i = R.id.line_room_img_exit;
                                                    ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                    if (imageView4 != null) {
                                                        i = R.id.line_room_img_mode;
                                                        ImageView imageView5 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                        if (imageView5 != null) {
                                                            i = R.id.line_room_img_owner_head;
                                                            CircleImageView circleImageView = (CircleImageView) ViewBindings.findChildViewById(view, i);
                                                            if (circleImageView != null) {
                                                                i = R.id.line_room_img_room_public;
                                                                ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                if (imageView6 != null) {
                                                                    i = R.id.line_room_img_sound;
                                                                    ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                    if (imageView7 != null) {
                                                                        i = R.id.line_room_img_start;
                                                                        ImageView imageView8 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                        if (imageView8 != null) {
                                                                            i = R.id.line_room_img_user_crown;
                                                                            ImageView imageView9 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                            if (imageView9 != null) {
                                                                                i = R.id.line_room_img_user_invite;
                                                                                ImageView imageView10 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                if (imageView10 != null) {
                                                                                    i = R.id.line_room_img_voice;
                                                                                    ImageView imageView11 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                    if (imageView11 != null) {
                                                                                        i = R.id.line_room_layout_bg;
                                                                                        ImageView imageView12 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                        if (imageView12 != null) {
                                                                                            i = R.id.line_room_layout_max_speed;
                                                                                            FrameLayout frameLayout5 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                            if (frameLayout5 != null) {
                                                                                                i = R.id.line_room_layout_max_vibrate;
                                                                                                FrameLayout frameLayout6 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                if (frameLayout6 != null) {
                                                                                                    i = R.id.line_room_layout_message;
                                                                                                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                                    if (linearLayout != null) {
                                                                                                        i = R.id.line_room_layout_sms_info;
                                                                                                        LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                                        if (linearLayout2 != null) {
                                                                                                            i = R.id.line_room_layout_speed_info;
                                                                                                            RelativeLayout relativeLayout = (RelativeLayout) ViewBindings.findChildViewById(view, i);
                                                                                                            if (relativeLayout != null) {
                                                                                                                i = R.id.line_room_layout_user_count;
                                                                                                                LinearLayout linearLayout3 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                if (linearLayout3 != null) {
                                                                                                                    i = R.id.line_room_layout_user_info;
                                                                                                                    RelativeLayout relativeLayout2 = (RelativeLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                    if (relativeLayout2 != null) {
                                                                                                                        i = R.id.line_room_layout_vibrate_info;
                                                                                                                        RelativeLayout relativeLayout3 = (RelativeLayout) ViewBindings.findChildViewById(view, i);
                                                                                                                        if (relativeLayout3 != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.line_room_line))) != null) {
                                                                                                                            i = R.id.line_room_playView;
                                                                                                                            PlayView_Four playView_Four = (PlayView_Four) ViewBindings.findChildViewById(view, i);
                                                                                                                            if (playView_Four != null) {
                                                                                                                                i = R.id.line_room_playView_vibrate;
                                                                                                                                PlayView_Four playView_Four2 = (PlayView_Four) ViewBindings.findChildViewById(view, i);
                                                                                                                                if (playView_Four2 != null) {
                                                                                                                                    i = R.id.line_room_progress_speed;
                                                                                                                                    VerticalProgressBar verticalProgressBar = (VerticalProgressBar) ViewBindings.findChildViewById(view, i);
                                                                                                                                    if (verticalProgressBar != null) {
                                                                                                                                        i = R.id.line_room_progress_vibrate;
                                                                                                                                        VerticalProgressBar verticalProgressBar2 = (VerticalProgressBar) ViewBindings.findChildViewById(view, i);
                                                                                                                                        if (verticalProgressBar2 != null) {
                                                                                                                                            i = R.id.line_room_recycler_message;
                                                                                                                                            RecyclerView recyclerView = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                                                                            if (recyclerView != null) {
                                                                                                                                                i = R.id.line_room_recycler_user;
                                                                                                                                                RecyclerView recyclerView2 = (RecyclerView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                if (recyclerView2 != null) {
                                                                                                                                                    i = R.id.line_room_txt_max_speed;
                                                                                                                                                    TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                    if (textView != null) {
                                                                                                                                                        i = R.id.line_room_txt_max_vibrate;
                                                                                                                                                        TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                        if (textView2 != null) {
                                                                                                                                                            i = R.id.line_room_txt_min_speed;
                                                                                                                                                            TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                            if (textView3 != null) {
                                                                                                                                                                i = R.id.line_room_txt_min_vibrate;
                                                                                                                                                                TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                if (textView4 != null) {
                                                                                                                                                                    i = R.id.line_room_txt_owner_name;
                                                                                                                                                                    TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                    if (textView5 != null) {
                                                                                                                                                                        i = R.id.line_room_txt_room_id;
                                                                                                                                                                        TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                        if (textView6 != null) {
                                                                                                                                                                            i = R.id.line_room_txt_speed;
                                                                                                                                                                            TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                            if (textView7 != null) {
                                                                                                                                                                                i = R.id.line_room_txt_user_count;
                                                                                                                                                                                TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                if (textView8 != null) {
                                                                                                                                                                                    i = R.id.line_room_txt_vibrate;
                                                                                                                                                                                    TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                    if (textView9 != null) {
                                                                                                                                                                                        i = R.id.progress_position;
                                                                                                                                                                                        ProgressBar progressBar = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                        if (progressBar != null) {
                                                                                                                                                                                            i = R.id.progress_speed;
                                                                                                                                                                                            ProgressBar progressBar2 = (ProgressBar) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                            if (progressBar2 != null) {
                                                                                                                                                                                                i = R.id.seek_smooth;
                                                                                                                                                                                                CustomSeekbar customSeekbar = (CustomSeekbar) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                if (customSeekbar != null) {
                                                                                                                                                                                                    i = R.id.txt_position;
                                                                                                                                                                                                    TextView textView10 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                    if (textView10 != null) {
                                                                                                                                                                                                        i = R.id.txt_position_label;
                                                                                                                                                                                                        TextView textView11 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                        if (textView11 != null) {
                                                                                                                                                                                                            i = R.id.txt_position_max;
                                                                                                                                                                                                            AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                            if (appCompatTextView != null) {
                                                                                                                                                                                                                i = R.id.txt_position_min;
                                                                                                                                                                                                                AppCompatTextView appCompatTextView2 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                if (appCompatTextView2 != null) {
                                                                                                                                                                                                                    i = R.id.txt_smooth;
                                                                                                                                                                                                                    TextView textView12 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                    if (textView12 != null) {
                                                                                                                                                                                                                        i = R.id.txt_smooth_label;
                                                                                                                                                                                                                        TextView textView13 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                        if (textView13 != null) {
                                                                                                                                                                                                                            i = R.id.txt_speed;
                                                                                                                                                                                                                            TextView textView14 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                            if (textView14 != null) {
                                                                                                                                                                                                                                i = R.id.txt_speed_label;
                                                                                                                                                                                                                                TextView textView15 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                                if (textView15 != null) {
                                                                                                                                                                                                                                    i = R.id.txt_speed_max;
                                                                                                                                                                                                                                    AppCompatTextView appCompatTextView3 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                                    if (appCompatTextView3 != null && (findChildViewById2 = ViewBindings.findChildViewById(view, (i = R.id.view_network_error))) != null) {
                                                                                                                                                                                                                                        ViewNetworkErrorBinding bind = ViewNetworkErrorBinding.bind(findChildViewById2);
                                                                                                                                                                                                                                        i = R.id.web_view;
                                                                                                                                                                                                                                        CustomWebView customWebView = (CustomWebView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                                        if (customWebView != null) {
                                                                                                                                                                                                                                            i = R.id.web_view_rocket;
                                                                                                                                                                                                                                            CustomWebView customWebView2 = (CustomWebView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                                            if (customWebView2 != null) {
                                                                                                                                                                                                                                                return new ActivityLineRoom2Binding((LinearLayout) view, button, imageView, imageView2, frameLayout, frameLayout2, constraintLayout, frameLayout3, constraintLayout2, editText, frameLayout4, imageView3, imageView4, imageView5, circleImageView, imageView6, imageView7, imageView8, imageView9, imageView10, imageView11, imageView12, frameLayout5, frameLayout6, linearLayout, linearLayout2, relativeLayout, linearLayout3, relativeLayout2, relativeLayout3, findChildViewById, playView_Four, playView_Four2, verticalProgressBar, verticalProgressBar2, recyclerView, recyclerView2, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8, textView9, progressBar, progressBar2, customSeekbar, textView10, textView11, appCompatTextView, appCompatTextView2, textView12, textView13, textView14, textView15, appCompatTextView3, bind, customWebView, customWebView2);
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

    public static ActivityLineRoom2Binding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityLineRoom2Binding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_line_room_2, viewGroup, false);
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
