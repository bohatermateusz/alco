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
import androidx.constraintlayout.widget.Guideline;
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
public final class ActivityLineRoomBinding implements ViewBinding {
    public final Button btnMessageSwitch;
    public final View guideView;
    public final ImageView imgPosition;
    public final ImageView imgRunMode;
    public final ConstraintLayout layoutPositionMax;
    public final ConstraintLayout layoutPositionMin;
    public final ConstraintLayout layoutSmsTouchPosition;
    public final FrameLayout layoutSpeedMax;
    public final ConstraintLayout layoutTouchPosition;
    public final Guideline lineMaxStart;
    public final Guideline lineMinStart;
    public final EditText lineRoomEditMessage;
    public final FrameLayout lineRoomFragmentPreview;
    public final ImageView lineRoomImgChat;
    public final ImageView lineRoomImgExit;
    public final ImageView lineRoomImgMode;
    public final CircleImageView lineRoomImgOwnerHead;
    public final ImageView lineRoomImgPause;
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
    public final View lineRoomMessageResizeHandle;
    public final PlayView_Four lineRoomPlayView;
    public final PlayView_Four lineRoomPlayViewVibrate;
    public final VerticalProgressBar lineRoomProgressSpeed;
    public final VerticalProgressBar lineRoomProgressVibrate;
    public final RecyclerView lineRoomRecyclerMessage;
    public final RecyclerView lineRoomRecyclerUser;
    public final TextView lineRoomTxtMaxSpeed;
    public final TextView lineRoomTxtMaxVibrate;
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

    private ActivityLineRoomBinding(LinearLayout linearLayout, Button button, View view, ImageView imageView, ImageView imageView2, ConstraintLayout constraintLayout, ConstraintLayout constraintLayout2, ConstraintLayout constraintLayout3, FrameLayout frameLayout, ConstraintLayout constraintLayout4, Guideline guideline, Guideline guideline2, EditText editText, FrameLayout frameLayout2, ImageView imageView3, ImageView imageView4, ImageView imageView5, CircleImageView circleImageView, ImageView imageView6, ImageView imageView7, ImageView imageView8, ImageView imageView9, ImageView imageView10, ImageView imageView11, ImageView imageView12, ImageView imageView13, FrameLayout frameLayout3, FrameLayout frameLayout4, LinearLayout linearLayout2, LinearLayout linearLayout3, RelativeLayout relativeLayout, LinearLayout linearLayout4, RelativeLayout relativeLayout2, RelativeLayout relativeLayout3, View view2, View view3, PlayView_Four playView_Four, PlayView_Four playView_Four2, VerticalProgressBar verticalProgressBar, VerticalProgressBar verticalProgressBar2, RecyclerView recyclerView, RecyclerView recyclerView2, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, ProgressBar progressBar, ProgressBar progressBar2, CustomSeekbar customSeekbar, TextView textView8, TextView textView9, AppCompatTextView appCompatTextView, AppCompatTextView appCompatTextView2, TextView textView10, TextView textView11, TextView textView12, TextView textView13, AppCompatTextView appCompatTextView3, ViewNetworkErrorBinding viewNetworkErrorBinding, CustomWebView customWebView, CustomWebView customWebView2) {
        this.rootView = linearLayout;
        this.btnMessageSwitch = button;
        this.guideView = view;
        this.imgPosition = imageView;
        this.imgRunMode = imageView2;
        this.layoutPositionMax = constraintLayout;
        this.layoutPositionMin = constraintLayout2;
        this.layoutSmsTouchPosition = constraintLayout3;
        this.layoutSpeedMax = frameLayout;
        this.layoutTouchPosition = constraintLayout4;
        this.lineMaxStart = guideline;
        this.lineMinStart = guideline2;
        this.lineRoomEditMessage = editText;
        this.lineRoomFragmentPreview = frameLayout2;
        this.lineRoomImgChat = imageView3;
        this.lineRoomImgExit = imageView4;
        this.lineRoomImgMode = imageView5;
        this.lineRoomImgOwnerHead = circleImageView;
        this.lineRoomImgPause = imageView6;
        this.lineRoomImgRoomPublic = imageView7;
        this.lineRoomImgSound = imageView8;
        this.lineRoomImgStart = imageView9;
        this.lineRoomImgUserCrown = imageView10;
        this.lineRoomImgUserInvite = imageView11;
        this.lineRoomImgVoice = imageView12;
        this.lineRoomLayoutBg = imageView13;
        this.lineRoomLayoutMaxSpeed = frameLayout3;
        this.lineRoomLayoutMaxVibrate = frameLayout4;
        this.lineRoomLayoutMessage = linearLayout2;
        this.lineRoomLayoutSmsInfo = linearLayout3;
        this.lineRoomLayoutSpeedInfo = relativeLayout;
        this.lineRoomLayoutUserCount = linearLayout4;
        this.lineRoomLayoutUserInfo = relativeLayout2;
        this.lineRoomLayoutVibrateInfo = relativeLayout3;
        this.lineRoomLine = view2;
        this.lineRoomMessageResizeHandle = view3;
        this.lineRoomPlayView = playView_Four;
        this.lineRoomPlayViewVibrate = playView_Four2;
        this.lineRoomProgressSpeed = verticalProgressBar;
        this.lineRoomProgressVibrate = verticalProgressBar2;
        this.lineRoomRecyclerMessage = recyclerView;
        this.lineRoomRecyclerUser = recyclerView2;
        this.lineRoomTxtMaxSpeed = textView;
        this.lineRoomTxtMaxVibrate = textView2;
        this.lineRoomTxtOwnerName = textView3;
        this.lineRoomTxtRoomId = textView4;
        this.lineRoomTxtSpeed = textView5;
        this.lineRoomTxtUserCount = textView6;
        this.lineRoomTxtVibrate = textView7;
        this.progressPosition = progressBar;
        this.progressSpeed = progressBar2;
        this.seekSmooth = customSeekbar;
        this.txtPosition = textView8;
        this.txtPositionLabel = textView9;
        this.txtPositionMax = appCompatTextView;
        this.txtPositionMin = appCompatTextView2;
        this.txtSmooth = textView10;
        this.txtSmoothLabel = textView11;
        this.txtSpeed = textView12;
        this.txtSpeedLabel = textView13;
        this.txtSpeedMax = appCompatTextView3;
        this.viewNetworkError = viewNetworkErrorBinding;
        this.webView = customWebView;
        this.webViewRocket = customWebView2;
    }

    public static ActivityLineRoomBinding bind(View view) {
        View findChildViewById;
        View findChildViewById2;
        View findChildViewById3;
        View findChildViewById4;
        int i = R.id.btn_message_switch;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null && (findChildViewById = ViewBindings.findChildViewById(view, (i = R.id.guide_view))) != null) {
            i = R.id.img_position;
            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
            if (imageView != null) {
                i = R.id.img_run_mode;
                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                if (imageView2 != null) {
                    i = R.id.layout_position_max;
                    ConstraintLayout constraintLayout = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                    if (constraintLayout != null) {
                        i = R.id.layout_position_min;
                        ConstraintLayout constraintLayout2 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                        if (constraintLayout2 != null) {
                            i = R.id.layout_sms_touch_position;
                            ConstraintLayout constraintLayout3 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                            if (constraintLayout3 != null) {
                                i = R.id.layout_speed_max;
                                FrameLayout frameLayout = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                if (frameLayout != null) {
                                    i = R.id.layout_touch_position;
                                    ConstraintLayout constraintLayout4 = (ConstraintLayout) ViewBindings.findChildViewById(view, i);
                                    if (constraintLayout4 != null) {
                                        i = R.id.line_max_start;
                                        Guideline guideline = (Guideline) ViewBindings.findChildViewById(view, i);
                                        if (guideline != null) {
                                            i = R.id.line_min_start;
                                            Guideline guideline2 = (Guideline) ViewBindings.findChildViewById(view, i);
                                            if (guideline2 != null) {
                                                i = R.id.line_room_edit_message;
                                                EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                                                if (editText != null) {
                                                    i = R.id.line_room_fragment_preview;
                                                    FrameLayout frameLayout2 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                    if (frameLayout2 != null) {
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
                                                                        i = R.id.line_room_img_pause;
                                                                        ImageView imageView6 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                        if (imageView6 != null) {
                                                                            i = R.id.line_room_img_room_public;
                                                                            ImageView imageView7 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                            if (imageView7 != null) {
                                                                                i = R.id.line_room_img_sound;
                                                                                ImageView imageView8 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                if (imageView8 != null) {
                                                                                    i = R.id.line_room_img_start;
                                                                                    ImageView imageView9 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                    if (imageView9 != null) {
                                                                                        i = R.id.line_room_img_user_crown;
                                                                                        ImageView imageView10 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                        if (imageView10 != null) {
                                                                                            i = R.id.line_room_img_user_invite;
                                                                                            ImageView imageView11 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                            if (imageView11 != null) {
                                                                                                i = R.id.line_room_img_voice;
                                                                                                ImageView imageView12 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                                if (imageView12 != null) {
                                                                                                    i = R.id.line_room_layout_bg;
                                                                                                    ImageView imageView13 = (ImageView) ViewBindings.findChildViewById(view, i);
                                                                                                    if (imageView13 != null) {
                                                                                                        i = R.id.line_room_layout_max_speed;
                                                                                                        FrameLayout frameLayout3 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                        if (frameLayout3 != null) {
                                                                                                            i = R.id.line_room_layout_max_vibrate;
                                                                                                            FrameLayout frameLayout4 = (FrameLayout) ViewBindings.findChildViewById(view, i);
                                                                                                            if (frameLayout4 != null) {
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
                                                                                                                                    if (relativeLayout3 != null && (findChildViewById2 = ViewBindings.findChildViewById(view, (i = R.id.line_room_line))) != null && (findChildViewById3 = ViewBindings.findChildViewById(view, (i = R.id.line_room_message_resize_handle))) != null) {
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
                                                                                                                                                                        i = R.id.line_room_txt_owner_name;
                                                                                                                                                                        TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                        if (textView3 != null) {
                                                                                                                                                                            i = R.id.line_room_txt_room_id;
                                                                                                                                                                            TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                            if (textView4 != null) {
                                                                                                                                                                                i = R.id.line_room_txt_speed;
                                                                                                                                                                                TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                if (textView5 != null) {
                                                                                                                                                                                    i = R.id.line_room_txt_user_count;
                                                                                                                                                                                    TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                    if (textView6 != null) {
                                                                                                                                                                                        i = R.id.line_room_txt_vibrate;
                                                                                                                                                                                        TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                        if (textView7 != null) {
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
                                                                                                                                                                                                        TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                        if (textView8 != null) {
                                                                                                                                                                                                            i = R.id.txt_position_label;
                                                                                                                                                                                                            TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                            if (textView9 != null) {
                                                                                                                                                                                                                i = R.id.txt_position_max;
                                                                                                                                                                                                                AppCompatTextView appCompatTextView = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                if (appCompatTextView != null) {
                                                                                                                                                                                                                    i = R.id.txt_position_min;
                                                                                                                                                                                                                    AppCompatTextView appCompatTextView2 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                    if (appCompatTextView2 != null) {
                                                                                                                                                                                                                        i = R.id.txt_smooth;
                                                                                                                                                                                                                        TextView textView10 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                        if (textView10 != null) {
                                                                                                                                                                                                                            i = R.id.txt_smooth_label;
                                                                                                                                                                                                                            TextView textView11 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                            if (textView11 != null) {
                                                                                                                                                                                                                                i = R.id.txt_speed;
                                                                                                                                                                                                                                TextView textView12 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                                if (textView12 != null) {
                                                                                                                                                                                                                                    i = R.id.txt_speed_label;
                                                                                                                                                                                                                                    TextView textView13 = (TextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                                    if (textView13 != null) {
                                                                                                                                                                                                                                        i = R.id.txt_speed_max;
                                                                                                                                                                                                                                        AppCompatTextView appCompatTextView3 = (AppCompatTextView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                                        if (appCompatTextView3 != null && (findChildViewById4 = ViewBindings.findChildViewById(view, (i = R.id.view_network_error))) != null) {
                                                                                                                                                                                                                                            ViewNetworkErrorBinding bind = ViewNetworkErrorBinding.bind(findChildViewById4);
                                                                                                                                                                                                                                            i = R.id.web_view;
                                                                                                                                                                                                                                            CustomWebView customWebView = (CustomWebView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                                            if (customWebView != null) {
                                                                                                                                                                                                                                                i = R.id.web_view_rocket;
                                                                                                                                                                                                                                                CustomWebView customWebView2 = (CustomWebView) ViewBindings.findChildViewById(view, i);
                                                                                                                                                                                                                                                if (customWebView2 != null) {
                                                                                                                                                                                                                                                    return new ActivityLineRoomBinding((LinearLayout) view, button, findChildViewById, imageView, imageView2, constraintLayout, constraintLayout2, constraintLayout3, frameLayout, constraintLayout4, guideline, guideline2, editText, frameLayout2, imageView3, imageView4, imageView5, circleImageView, imageView6, imageView7, imageView8, imageView9, imageView10, imageView11, imageView12, imageView13, frameLayout3, frameLayout4, linearLayout, linearLayout2, relativeLayout, linearLayout3, relativeLayout2, relativeLayout3, findChildViewById2, findChildViewById3, playView_Four, playView_Four2, verticalProgressBar, verticalProgressBar2, recyclerView, recyclerView2, textView, textView2, textView3, textView4, textView5, textView6, textView7, progressBar, progressBar2, customSeekbar, textView8, textView9, appCompatTextView, appCompatTextView2, textView10, textView11, textView12, textView13, appCompatTextView3, bind, customWebView, customWebView2);
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
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static ActivityLineRoomBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityLineRoomBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_line_room, viewGroup, false);
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
