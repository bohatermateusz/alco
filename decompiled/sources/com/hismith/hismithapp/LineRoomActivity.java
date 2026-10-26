package com.hismith.hismithapp;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ValueAnimator;
import android.content.BroadcastReceiver;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.graphics.drawable.Drawable;
import android.media.AudioManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import android.view.animation.LinearInterpolator;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.activity.OnBackPressedCallback;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.Guideline;
import androidx.core.content.ContextCompat;
import androidx.exifinterface.media.ExifInterface;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.facebook.internal.NativeProtocol;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.hismith.hismithapp.adapter.CircleImgAdapter;
import com.hismith.hismithapp.adapter.LineRoomMessageAdapter;
import com.hismith.hismithapp.adapter.LineUserAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.dialog.PagerDrawerPopup;
import com.hismith.hismithapp.dialog.UserDrawerPopupDialog;
import com.hismith.hismithapp.dialog.UserInfoDrawerPopupDialog;
import com.hismith.hismithapp.floating.FloatBase;
import com.hismith.hismithapp.floating.FloatManager;
import com.hismith.hismithapp.fragment.PreviewFragment2;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.AudioSocketManager;
import com.hismith.hismithapp.manager.LineAudioManager;
import com.hismith.hismithapp.model.LineRoomInfoModel;
import com.hismith.hismithapp.model.ProductModel;
import com.hismith.hismithapp.model.RemotePlayModel;
import com.hismith.hismithapp.model.socket.PlayLineSocketModel;
import com.hismith.hismithapp.model.socket.SocketUserModel;
import com.hismith.hismithapp.observer.AppBackgroundUtil;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.receiver.NetBroadcastReceiver;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.CheckUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.IntentUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.hismith.hismithapp.view.CustomSeekbar;
import com.hismith.hismithapp.view.CustomWebView;
import com.hismith.hismithapp.view.PlayView_Four;
import com.hismith.hismithapp.view.VerticalProgressBar;
import com.lxj.xpopup.XPopup;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import okhttp3.Response;

/* loaded from: classes3.dex */
public class LineRoomActivity extends BaseActivity implements SocketManager.OnSocketNotifyListener, AudioSocketManager.OnAudioSocketNotifyListener, View.OnClickListener, PlayView_Four.OnLineChangeListener, NetBroadcastReceiver.NetChangeListener, LineAudioManager.OnAudioSpeedChangeListener, View.OnTouchListener {
    private static final int MESSAGE_DEFAULT_HEIGHT_DP = 120;
    private static final float MESSAGE_MAX_SIZE_HEIGHT = 0.85f;
    private static final float MESSAGE_MAX_SIZE_WIDTH = 0.95f;
    private static final int MESSAGE_MIN_HEIGHT_DP = 59;
    private static final int MESSAGE_MIN_WIDTH_DP = 200;
    private static final int WHAT_SOCKET_FAILURE = 50;
    private static final int WHAT_SOCKET_OPEN = 49;
    private static final int WHAT_SOCKET_SNYC_PLAY = 51;
    private static final int mDefaultReconnCount = 0;
    private float downVibrateY;
    private float downX;
    private float downY;
    private float endDownX;
    private UserInfoDrawerPopupDialog hallUserInfoDrawerPopupDialog;
    private ImageView img_position;
    private ImageView img_run_mode;
    private String isRoomVoice;
    private ViewGroup layout_position_max;
    private ViewGroup layout_position_min;
    private ConstraintLayout layout_sms_touch_position;
    private ViewGroup layout_speed_max;
    private ConstraintLayout layout_touch_position;
    private Guideline line_max_start;
    private Guideline line_min_start;
    private AudioManager mAudioManager;
    private EditText mEditMessage;
    private FloatBase mFloatBase;
    private List<SocketUserModel> mHallUserList;
    private CircleImgAdapter mHeadAdapter;
    private ImageView mImgChat;
    private ImageView mImgCrown;
    private ImageView mImgExit;
    private ImageView mImgMode;
    private ImageView mImgNetworkError;
    private CircleImageView mImgOwnerHead;
    private ImageView mImgRoomPublic;
    private ImageView mImgSound;
    private ImageView mImgStart;
    private ImageView mImgUserInvite;
    private ImageView mImgVoice;
    private ImageView mLayoutBg;
    private ViewGroup mLayoutMaxSpeed;
    private ViewGroup mLayoutMaxVibrate;
    private LinearLayout mLayoutSMSInfo;
    private RelativeLayout mLayoutSpeedInfo;
    private LinearLayout mLayoutUserCount;
    private RelativeLayout mLayoutVibrateInfo;
    private LineRoomMessageAdapter mMessageAdapter;
    private ValueAnimator mMessageLayoutAnimator;
    private LinearLayoutManager mMessageLayoutManager;
    private View mMessageResizeHandle;
    private List<PlayLineSocketModel> mMessages;
    private PagerDrawerPopup mModePagerDialog;
    private NetBroadcastReceiver mNetBroadcastReceiver;
    private PlayView_Four mPlayView;
    private PlayView_Four mPlayViewVibrate;
    private int mPosition;
    private PreviewFragment2 mPreviewFragment;
    private ProgressBar mProgressSpeed;
    private VerticalProgressBar mProgressVibrate;
    private RecyclerView mRecyclerMessage;
    private int mRecyclerMessageHeight;
    private RecyclerView mRecyclerUser;
    private float mResizeDownRawX;
    private float mResizeDownRawY;
    private int mResizeStartHeight;
    private int mResizeStartWidth;
    private List<SocketUserModel> mRoomUserList;
    private TextView mTxtMaxSpeed;
    private TextView mTxtMaxVibrate;
    private TextView mTxtNetworkError;
    private TextView mTxtOwnerName;
    private TextView mTxtRoomId;
    private TextView mTxtSpeed;
    private TextView mTxtUserCount;
    private TextView mTxtVibrate;
    private LinearLayoutManager mUserLayoutManager;
    private List<LineRoomInfoModel.RoomInfoUserModel> mUsers;
    private View mViewLine;
    private View mViewNetworkError;
    private FrameLayout mViewPreview;
    private CustomWebView mWebView;
    private CustomWebView mWebViewRocket;
    private List<PlayLineSocketModel> mWorldMessage;
    private LineRoomMessageAdapter mWorldMessageAdapter;
    private ProgressBar progress_position;
    private ProgressBar progress_speed;
    private UserDrawerPopupDialog rightHallUserDialog;
    private UserDrawerPopupDialog rightRoomUserDialog;
    private UserInfoDrawerPopupDialog roomUserInfoDrawerPopupDialog;
    private CustomSeekbar seek_smooth;
    private float startDownX;
    private TextView txt_position;
    private TextView txt_position_label;
    private TextView txt_position_max;
    private TextView txt_position_min;
    private TextView txt_smooth;
    private TextView txt_smooth_label;
    private TextView txt_speed;
    private TextView txt_speed_label;
    private TextView txt_speed_max;
    private String FRAGMENT_PREVIEW_TAG = "fragment_preview_tag";
    private boolean isModeing = false;
    private int mModeingId = -1;
    private int mLastNetStatus = -1;
    private int isRoomOrWorld = 0;
    private int mSafeMax = 100;
    private int mSafeVibrateMax = 100;
    private boolean isSafeTouch = true;
    private boolean isShowMessage = true;
    private int animationDuration = 600;
    private int lastSpeed = 0;
    private int lastVibrate = 0;
    private int recePosition = 0;
    private String captainId = "";
    private boolean isSync = false;
    private final int mRocketDuration = 1000;
    private boolean isCloseSpeaker = false;
    private boolean isCloseMic = true;
    private boolean runMode = false;
    private boolean runPisitionQiu = false;
    private int mReconnCount = 0;
    private int mAudioReconnCount = 0;
    private int mSocketFailureDelay = 3000;
    private int mDisconnType = 0;
    private final int REQUEST_SMALL = 101;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.LineRoomActivity.1
        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int i = message.what;
            Bundle data = message.getData();
            String string = data != null ? data.getString(Constans.BleType) : "";
            if (i == 1) {
                int intValue = ((Integer) message.obj).intValue();
                if (intValue == 160) {
                    LineRoomActivity.this.runMode = true;
                    LineRoomActivity.this.runPisitionQiu = false;
                    LineRoomActivity.this.img_run_mode.setImageResource(R.mipmap.ic_auto_switch_on);
                    LineRoomActivity.this.layout_touch_position.setVisibility(8);
                    if (MyApplication.getApplication().getmUserModel() != null && LineRoomActivity.this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
                        LineRoomActivity.this.mPlayView.setTouch(true);
                        LineRoomActivity.this.mPlayViewVibrate.setTouch(true);
                        SocketManager.getInstance().setStrokeType(0);
                    }
                } else if (intValue == 161) {
                    LineRoomActivity.this.runMode = false;
                    LineRoomActivity.this.runPisitionQiu = true;
                    LineRoomActivity.this.img_run_mode.setImageResource(R.mipmap.ic_auto_switch_off);
                    if (MyApplication.getApplication().getmUserModel() != null && LineRoomActivity.this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
                        LineRoomActivity.this.layout_touch_position.setVisibility(0);
                    }
                    LineRoomActivity.this.mPlayView.setTouch(false);
                    LineRoomActivity.this.mPlayViewVibrate.setTouch(false);
                    LineRoomActivity.this.manualTouchPosition();
                    if (MyApplication.getApplication().getmUserModel() != null && LineRoomActivity.this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
                        SocketManager.getInstance().setStrokeType(1);
                    }
                }
            } else if (i == 49) {
                LineRoomActivity.this.mViewNetworkError.setVisibility(8);
                LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                LanguageUtil.setText(lineRoomActivity, lineRoomActivity.mTxtNetworkError, "notNetwork");
            } else if (i != 50) {
                switch (i) {
                    case 3:
                        int intValue2 = ((Integer) message.obj).intValue();
                        LineRoomActivity.this.mProgressSpeed.setProgress(intValue2);
                        LineRoomActivity.this.mTxtSpeed.setText(intValue2 + "");
                        LineRoomActivity.this.setFloatSpeed(intValue2);
                        if (MyApplication.getApplication().getmUserModel() != null && LineRoomActivity.this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "") && "CA".equals(string)) {
                            LineRoomActivity.this.lastSpeed = intValue2;
                            SocketManager.getInstance().setSpeed(intValue2, (JsonObject) null);
                            break;
                        }
                        break;
                    case 4:
                        int intValue3 = ((Integer) message.obj).intValue();
                        if (LineRoomActivity.this.mFloatBase != null) {
                            LineRoomActivity.this.mFloatBase.refreshFloatModeUI(intValue3);
                            break;
                        }
                        break;
                    case 5:
                        int intValue4 = ((Integer) message.obj).intValue();
                        if (intValue4 != LineRoomActivity.this.mProgressVibrate.getProgress() || intValue4 == 0) {
                            LineRoomActivity.this.mProgressVibrate.setProgress(intValue4);
                            LineRoomActivity.this.mTxtVibrate.setText(intValue4 + "");
                            LineRoomActivity.this.setFloatVibrate(intValue4);
                            if (MyApplication.getApplication().getmUserModel() != null && LineRoomActivity.this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "") && "CA".equals(string)) {
                                LineRoomActivity.this.lastVibrate = intValue4;
                                SocketManager.getInstance().setVibrate(intValue4, (JsonObject) null);
                                break;
                            }
                        }
                        break;
                    case 6:
                        if (LineRoomActivity.this.mFloatBase != null) {
                            LineRoomActivity.this.mFloatBase.refreshFloatVibrateModeUI(((Integer) message.obj).intValue());
                            break;
                        }
                        break;
                    case 7:
                        LineRoomActivity.this.refreshStartBitUI(((Integer) message.obj).intValue());
                        break;
                    case 8:
                        LineRoomActivity.this.refreshEndBitUI(((Integer) message.obj).intValue());
                        break;
                    case 9:
                        LineRoomActivity.this.refreshSmoothUI(((Integer) message.obj).intValue());
                        break;
                    case 10:
                        LineRoomActivity.this.refreshPositionUI(((Integer) message.obj).intValue());
                        if (LineRoomActivity.this.runPisitionQiu) {
                            LineRoomActivity.this.manualTouchPosition();
                            break;
                        }
                        break;
                }
            } else {
                LineRoomActivity.this.mViewNetworkError.setVisibility(0);
            }
            return false;
        }
    });
    private View.OnTouchListener vibrateSafeTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.LineRoomActivity.25
        /* JADX WARN: Code restructure failed: missing block: B:11:0x0017, code lost:
        
            if (r0 != 3) goto L42;
         */
        @Override // android.view.View.OnTouchListener
        /*
            Code decompiled incorrectly, please refer to instructions dump.
        */
        public boolean onTouch(View view, MotionEvent motionEvent) {
            if (!LineRoomActivity.this.isSafeTouch) {
                return false;
            }
            int action = motionEvent.getAction();
            if (action != 0) {
                if (action != 1) {
                    if (action == 2) {
                        int top = (int) (view.getTop() + (motionEvent.getY() - LineRoomActivity.this.downVibrateY));
                        if (top < 0) {
                            top = 0;
                        } else if (top > LineRoomActivity.this.mProgressVibrate.getHeight()) {
                            top = LineRoomActivity.this.mProgressVibrate.getHeight();
                        }
                        int height = view.getHeight() + top;
                        LogUtil.e("onTouch  t:" + top + ", b:" + height + ", offset:" + (height - top));
                        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) LineRoomActivity.this.mLayoutMaxVibrate.getLayoutParams();
                        layoutParams.setMargins(0, top, 0, 0);
                        LineRoomActivity.this.mLayoutMaxVibrate.setLayoutParams(layoutParams);
                        int height2 = 100 - (((int) (top * 100.0d)) / LineRoomActivity.this.mProgressVibrate.getHeight());
                        LineRoomActivity.this.mTxtMaxVibrate.setText(height2 + "");
                        LineRoomActivity.this.mSafeVibrateMax = height2;
                        LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                        SharedPreferencesUtil.put(lineRoomActivity, Constans.SAFE_FUN2, Integer.valueOf(lineRoomActivity.mSafeVibrateMax));
                        if (LineRoomActivity.this.mPreviewFragment != null) {
                            LineRoomActivity.this.mPreviewFragment.setmVibrateSafe(LineRoomActivity.this.mSafeVibrateMax);
                        }
                        if (height2 < 0 || height2 >= LineRoomActivity.this.lastVibrate) {
                            if (height2 >= LineRoomActivity.this.lastVibrate && LineRoomActivity.this.lastVibrate != LineRoomActivity.this.mProgressVibrate.getProgress() && !LineRoomActivity.this.isModeing) {
                                BleManager.getInstance(LineRoomActivity.this).writeVibrate(LineRoomActivity.this.lastVibrate);
                            }
                        } else if (height2 != LineRoomActivity.this.mProgressVibrate.getProgress()) {
                            if (MyApplication.getApplication().getmUserModel() == null || !LineRoomActivity.this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
                                BleManager.getInstance(LineRoomActivity.this).writeVibrate(height2);
                            } else {
                                LineRoomActivity.this.mPlayViewVibrate.onPointArrChange(height2, null);
                            }
                        }
                    }
                }
                LogUtil.e("onTouch 松开Y:" + motionEvent.getY());
            } else {
                LineRoomActivity.this.downVibrateY = motionEvent.getY();
            }
            return true;
        }
    };
    private View.OnTouchListener safeOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.LineRoomActivity.26
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            if (!LineRoomActivity.this.isSafeTouch) {
                return false;
            }
            int action = motionEvent.getAction();
            if (action == 0) {
                LineRoomActivity.this.downX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - LineRoomActivity.this.downX;
            int right = LineRoomActivity.this.progress_speed.getRight() + (LineRoomActivity.this.layout_speed_max.getWidth() / 2);
            LogUtil.e("getRight():" + LineRoomActivity.this.layout_speed_max.getRight() + ", width:" + LineRoomActivity.this.progress_speed.getRight() + ", " + LineRoomActivity.this.layout_speed_max.getWidth());
            int right2 = (int) (right - (LineRoomActivity.this.layout_speed_max.getRight() + x));
            if (right2 < 0) {
                right2 = 0;
            } else if (right2 > LineRoomActivity.this.progress_speed.getWidth()) {
                right2 = LineRoomActivity.this.progress_speed.getWidth();
            }
            LogUtil.e("safe right:" + right2);
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) LineRoomActivity.this.layout_speed_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right2, 0);
            LineRoomActivity.this.layout_speed_max.setLayoutParams(layoutParams);
            int width = (int) (100.0d - ((right2 * 100.0d) / LineRoomActivity.this.progress_speed.getWidth()));
            LineRoomActivity.this.mSafeMax = width;
            LineRoomActivity lineRoomActivity = LineRoomActivity.this;
            SharedPreferencesUtil.put(lineRoomActivity, Constans.SAFE_FUN1, Integer.valueOf(lineRoomActivity.mSafeMax));
            if (LineRoomActivity.this.mPreviewFragment != null) {
                LineRoomActivity.this.mPreviewFragment.setmSafe(LineRoomActivity.this.mSafeMax);
            }
            LineRoomActivity.this.txt_speed_max.setText(width + "");
            if (width < 0 || width >= LineRoomActivity.this.lastSpeed) {
                if (width < LineRoomActivity.this.lastSpeed || LineRoomActivity.this.lastSpeed == LineRoomActivity.this.progress_speed.getProgress() || LineRoomActivity.this.isModeing) {
                    return true;
                }
                BleManager.getInstance(LineRoomActivity.this).writeSpeed(LineRoomActivity.this.lastSpeed);
                return true;
            }
            if (width == LineRoomActivity.this.progress_speed.getProgress()) {
                return true;
            }
            if (MyApplication.getApplication().getmUserModel() == null || !LineRoomActivity.this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
                BleManager.getInstance(LineRoomActivity.this).writeSpeed(width);
                return true;
            }
            LineRoomActivity.this.mPlayView.onPointArrChange(width, null);
            return true;
        }
    };
    private int mPositionEnd = 100;
    private int start_stop_position_diff = Constans.start_stop_position_diff;
    private View.OnTouchListener endOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.LineRoomActivity.27
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                LineRoomActivity.this.endDownX = motionEvent.getX();
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - LineRoomActivity.this.endDownX;
            int width = LineRoomActivity.this.progress_position.getWidth();
            int right = (int) ((LineRoomActivity.this.progress_position.getRight() + (LineRoomActivity.this.layout_position_max.getWidth() / 2)) - (LineRoomActivity.this.layout_position_max.getRight() + x));
            if (right < 0) {
                right = 0;
            } else if (right > width) {
                right = width;
            }
            double d = width;
            int i = (int) (100.0d - ((right * 100.0d) / d));
            if (i <= LineRoomActivity.this.mPositionStart + LineRoomActivity.this.start_stop_position_diff) {
                int i2 = LineRoomActivity.this.mPositionStart + LineRoomActivity.this.start_stop_position_diff;
                i = i2 > 100 ? 100 : i2;
                right = (int) (d - ((d / 100.0d) * i));
            }
            int i3 = i <= 100 ? i < 0 ? 0 : i : 100;
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) LineRoomActivity.this.layout_position_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right, 0);
            LineRoomActivity.this.layout_position_max.setLayoutParams(layoutParams);
            LineRoomActivity.this.mPositionEnd = i3;
            LogUtil.e("结束位置：" + LineRoomActivity.this.mPositionEnd + ", right:" + right);
            LineRoomActivity.this.txt_position_max.setText(i3 + "");
            BleManager.getInstance(LineRoomActivity.this).writeStopPosition(LineRoomActivity.this.mPositionEnd);
            return true;
        }
    };
    private int mPositionStart = 0;
    private View.OnTouchListener startOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.LineRoomActivity.28
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                LineRoomActivity.this.startDownX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                LogUtil.e("onTouch left:" + view.getLeft());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - LineRoomActivity.this.startDownX;
            int width = LineRoomActivity.this.progress_position.getWidth();
            int left = (int) ((LineRoomActivity.this.layout_position_min.getLeft() + x) - (LineRoomActivity.this.progress_position.getLeft() - (LineRoomActivity.this.layout_position_min.getWidth() / 2)));
            LogUtil.e("onTouch moveX:" + x + ", left():" + left);
            if (left < 0) {
                left = 0;
            } else if (left > width) {
                left = width;
            }
            double d = width;
            int i = (int) ((left * 100.0d) / d);
            if (i >= LineRoomActivity.this.mPositionEnd - LineRoomActivity.this.start_stop_position_diff) {
                int i2 = LineRoomActivity.this.mPositionEnd - LineRoomActivity.this.start_stop_position_diff;
                i = i2 < 0 ? 0 : i2;
                left = (int) ((d / 100.0d) * i);
            }
            if (i > 100) {
                i = 100;
            } else if (i < 0) {
                i = 0;
            }
            ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) LineRoomActivity.this.layout_position_min.getLayoutParams();
            layoutParams.setMargins(left, 0, 0, 0);
            LineRoomActivity.this.layout_position_min.setLayoutParams(layoutParams);
            LineRoomActivity.this.mPositionStart = i;
            LogUtil.e("开始位置：" + LineRoomActivity.this.mPositionStart + ", left:" + left);
            LineRoomActivity.this.txt_position_min.setText(i + "");
            BleManager.getInstance(LineRoomActivity.this).writeStartPosition(LineRoomActivity.this.mPositionStart);
            return true;
        }
    };
    private View.OnTouchListener positionOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.LineRoomActivity.29
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            int i = 0;
            if (action == 0) {
                LineRoomActivity.this.runPisitionQiu = false;
            } else if (action != 2) {
                return true;
            }
            LineRoomActivity lineRoomActivity = LineRoomActivity.this;
            int changeUIPosition = lineRoomActivity.changeUIPosition(lineRoomActivity.layout_touch_position, LineRoomActivity.this.img_position, motionEvent.getX(), motionEvent.getY());
            if (changeUIPosition > 100) {
                i = 100;
            } else if (changeUIPosition >= 0) {
                i = changeUIPosition;
            }
            BleManager.getInstance(LineRoomActivity.this).writePosition(i);
            if (MyApplication.getApplication().getmUserModel() == null || !LineRoomActivity.this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
                return true;
            }
            SocketManager.getInstance().setStrokeNow(i);
            return true;
        }
    };
    private final View.OnTouchListener messageResizeTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.LineRoomActivity.34
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int actionMasked = motionEvent.getActionMasked();
            if (actionMasked != 0) {
                if (actionMasked != 1) {
                    if (actionMasked == 2) {
                        LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                        lineRoomActivity.resizeMessageLayout(lineRoomActivity.mResizeStartWidth + ((int) (motionEvent.getRawX() - LineRoomActivity.this.mResizeDownRawX)), LineRoomActivity.this.mResizeStartHeight - ((int) (motionEvent.getRawY() - LineRoomActivity.this.mResizeDownRawY)));
                        return true;
                    }
                    if (actionMasked != 3) {
                        return true;
                    }
                }
                if (LineRoomActivity.this.mMessageAdapter != null) {
                    LineRoomActivity.this.mMessageAdapter.notifyDataSetChanged();
                }
                return true;
            }
            if (LineRoomActivity.this.mMessageLayoutAnimator != null) {
                LineRoomActivity.this.mMessageLayoutAnimator.cancel();
            }
            LineRoomActivity.this.mResizeDownRawX = motionEvent.getRawX();
            LineRoomActivity.this.mResizeDownRawY = motionEvent.getRawY();
            LineRoomActivity lineRoomActivity2 = LineRoomActivity.this;
            lineRoomActivity2.mResizeStartWidth = lineRoomActivity2.mRecyclerMessage.getWidth();
            LineRoomActivity lineRoomActivity3 = LineRoomActivity.this;
            lineRoomActivity3.mResizeStartHeight = lineRoomActivity3.mRecyclerMessage.getHeight();
            if (LineRoomActivity.this.mResizeStartWidth <= 0) {
                LineRoomActivity lineRoomActivity4 = LineRoomActivity.this;
                lineRoomActivity4.mResizeStartWidth = lineRoomActivity4.mRecyclerMessage.getLayoutParams().width;
            }
            if (LineRoomActivity.this.mResizeStartHeight <= 0) {
                LineRoomActivity lineRoomActivity5 = LineRoomActivity.this;
                lineRoomActivity5.mResizeStartHeight = lineRoomActivity5.mRecyclerMessage.getLayoutParams().height;
            }
            return true;
        }
    };
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.LineRoomActivity.39
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(intent.getAction())) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                LogUtil.e("蓝牙广播数据接收：" + stringExtra + ", data0:" + stringExtra2 + ", data1:" + stringExtra3);
                if ("AA".equals(stringExtra)) {
                    LineRoomActivity.this.initBleAAData(stringExtra2, stringExtra3);
                } else if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    LineRoomActivity.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                }
            }
        }
    };

    class GameWebViewClient extends WebViewClient {
        GameWebViewClient() {
        }

        @Override // android.webkit.WebViewClient
        public void onPageFinished(WebView webView, String str) {
            super.onPageFinished(webView, str);
            LogUtil.e("mWebView.getContentHeight()" + LineRoomActivity.this.mWebView.getContentHeight());
            LogUtil.e(" * mWebView.getScale():" + LineRoomActivity.this.mWebView.getScale());
            LogUtil.e("mWebView.getHeight():" + LineRoomActivity.this.mWebView.getHeight());
        }

        @Override // android.webkit.WebViewClient
        public void onPageStarted(WebView webView, String str, Bitmap bitmap) {
            super.onPageStarted(webView, str, bitmap);
        }

        @Override // android.webkit.WebViewClient
        public boolean shouldOverrideUrlLoading(WebView webView, String str) {
            LogUtil.e("shouldOverrideUrlLoading url:" + str);
            try {
                if (!str.startsWith("http:") && !str.startsWith("https:")) {
                    AppUtil.startUriToWeb(LineRoomActivity.this, str);
                    return true;
                }
                webView.loadUrl(str);
            } catch (Exception unused) {
            }
            return true;
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public int changeUIPosition(View view, View view2, float f, float f2) {
        int width = (int) (f - (view2.getWidth() / 2.0f));
        int height = (int) (f2 - (view2.getHeight() / 2.0f));
        int width2 = view2.getWidth() + width;
        int height2 = view2.getHeight() + height;
        if (width < 0) {
            width2 = view2.getWidth();
            width = 0;
        }
        if (width2 > view.getWidth()) {
            width2 = view.getWidth();
            width = width2 - view2.getWidth();
        }
        if (height < 0) {
            height2 = view2.getHeight();
            height = 0;
        }
        if (height2 > view.getHeight()) {
            height2 = view.getHeight();
            height = height2 - view2.getHeight();
        }
        view2.layout(width, height, width2, height2);
        return 100 - ((int) (((height * 1.0f) / (this.layout_touch_position.getHeight() - view2.getHeight())) * 100.0f));
    }

    private boolean checkRecordPermission() {
        return ContextCompat.checkSelfPermission(this, "android.permission.RECORD_AUDIO") == 0;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void closePreviewFragment() {
        try {
            this.isModeing = false;
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.LineRoomActivity.12
                @Override // java.lang.Runnable
                public void run() {
                    LineRoomActivity.this.mViewPreview.setVisibility(8);
                    LineRoomActivity.this.mPreviewFragment = null;
                    Fragment findFragmentByTag = LineRoomActivity.this.getSupportFragmentManager().findFragmentByTag(LineRoomActivity.this.FRAGMENT_PREVIEW_TAG);
                    if (findFragmentByTag != null) {
                        LineRoomActivity.this.getSupportFragmentManager().beginTransaction().remove(findFragmentByTag).commitAllowingStateLoss();
                    }
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void destroyFloatWindow() {
        LogUtil.e("准备清除悬浮窗");
        if (this.mFloatBase != null) {
            LogUtil.e("清除悬浮窗");
            this.mFloatBase.destroy();
            this.mFloatBase = null;
        }
    }

    private void hideChatLayout() {
        ValueAnimator ofInt = ValueAnimator.ofInt(this.mRecyclerMessage.getHeight(), MeasureUtil.dp2px(this, 59.0f));
        this.mMessageLayoutAnimator = ofInt;
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.LineRoomActivity.36
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                int intValue = ((Integer) valueAnimator.getAnimatedValue()).intValue();
                LogUtil.e("onAnimationUpdate h:" + intValue);
                LineRoomActivity.this.mRecyclerMessage.getLayoutParams().height = intValue;
                LineRoomActivity.this.mRecyclerMessage.requestLayout();
            }
        });
        this.mMessageLayoutAnimator.addListener(new AnimatorListenerAdapter() { // from class: com.hismith.hismithapp.LineRoomActivity.37
            @Override // android.animation.AnimatorListenerAdapter, android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                super.onAnimationEnd(animator);
                LineRoomActivity.this.mRecyclerMessage.scrollToPosition(LineRoomActivity.this.mMessages.size() - 1);
            }
        });
        this.mMessageLayoutAnimator.setDuration(this.animationDuration);
        this.mMessageLayoutAnimator.start();
    }

    private void init() {
        this.mSafeMax = ((Integer) SharedPreferencesUtil.get(this, Constans.SAFE_FUN1, 40)).intValue();
        this.mSafeVibrateMax = ((Integer) SharedPreferencesUtil.get(this, Constans.SAFE_FUN2, 40)).intValue();
        this.isRoomVoice = (String) SharedPreferencesUtil.get(this, Constans.roomVoice);
        this.mAudioManager = (AudioManager) getSystemService("audio");
        this.mRecyclerMessageHeight = MeasureUtil.dip2px(this, 120.0f);
        this.mRoomUserList = new ArrayList();
        this.mHallUserList = new ArrayList();
        this.mPlayView.setProgressBar(this.mProgressSpeed);
        this.mPlayViewVibrate.setProgressBar(this.mProgressVibrate);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this, 0, false);
        this.mUserLayoutManager = linearLayoutManager;
        this.mRecyclerUser.setLayoutManager(linearLayoutManager);
        this.mUsers = new ArrayList();
        CircleImgAdapter circleImgAdapter = new CircleImgAdapter(this, this.mUsers);
        this.mHeadAdapter = circleImgAdapter;
        this.mRecyclerUser.setAdapter(circleImgAdapter);
        LinearLayoutManager linearLayoutManager2 = new LinearLayoutManager(this);
        this.mMessageLayoutManager = linearLayoutManager2;
        this.mRecyclerMessage.setLayoutManager(linearLayoutManager2);
        this.mMessages = new ArrayList();
        this.mWorldMessage = new ArrayList();
        this.mMessageAdapter = new LineRoomMessageAdapter(this, this.mMessages);
        this.mWorldMessageAdapter = new LineRoomMessageAdapter(this, this.mWorldMessage);
        this.mRecyclerMessage.setAdapter(this.mMessageAdapter);
        this.isRoomOrWorld = 0;
        this.mModePagerDialog = new PagerDrawerPopup(this);
        if (TextUtils.isEmpty(this.isRoomVoice) || "0".equals(this.isRoomVoice)) {
            this.isCloseSpeaker = true;
            this.mImgSound.setVisibility(8);
            this.isCloseMic = true;
            this.mImgVoice.setVisibility(8);
        } else {
            AudioSocketManager.getInstance().connSocket(3, this.isCloseSpeaker);
            AudioManager audioManager = this.mAudioManager;
            if (audioManager != null) {
                audioManager.requestAudioFocus(null, 3, 2);
            }
        }
        boolean z = this.isCloseSpeaker;
        ImageView imageView = this.mImgSound;
        if (z) {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_line_sound_close));
        } else {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_line_sound));
        }
        boolean z2 = this.isCloseMic;
        ImageView imageView2 = this.mImgVoice;
        if (z2) {
            imageView2.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_line_voice_close));
        } else {
            imageView2.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_line_voice));
        }
        initChatLayout();
        BleManager.getInstance(this).setWriteSpeed(false);
        if (SocketManager.getInstance().getRoomInfoModel() != null) {
            initRoomInfo(SocketManager.getInstance().getRoomInfoModel());
        }
        initWebView();
        initProductData();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleAAData(String str, String str2) {
        str.hashCode();
        switch (str) {
            case "03":
                try {
                    int intValue = Integer.valueOf(str2, 16).intValue();
                    if ((intValue != this.mProgressSpeed.getProgress() || intValue == 0) && MyApplication.getApplication().getmUserModel() != null && this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
                        this.lastSpeed = intValue;
                        SocketManager.getInstance().setSpeed(intValue, (JsonObject) null);
                        break;
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                    break;
                }
                break;
            case "04":
                break;
            case "05":
                if (this.mFloatBase != null) {
                    this.mFloatBase.refreshFloatModeUI(Integer.valueOf(str2, 16).intValue());
                    return;
                }
                return;
            case "06":
                if (MyApplication.getApplication().getmProductModel() == null || MyApplication.getApplication().getmProductModel().getClimax() == 1 || this.mFloatBase == null) {
                    return;
                }
                this.mFloatBase.refreshFloatVibrateModeUI(Integer.valueOf(str2, 16).intValue());
                return;
            default:
                return;
        }
        int intValue2 = Integer.valueOf(str2, 16).intValue();
        if (intValue2 != this.mProgressSpeed.getProgress() || intValue2 == 0) {
            Message message = new Message();
            message.what = 3;
            message.obj = Integer.valueOf(intValue2);
            this.mHandler.sendMessage(message);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleCCData(String str, String str2, String str3) {
        try {
            Message message = new Message();
            message.what = Integer.valueOf(str2, 16).intValue();
            message.obj = Integer.valueOf(str3, 16);
            Bundle bundle = new Bundle();
            bundle.putString(Constans.BleType, str);
            message.setData(bundle);
            this.mHandler.sendMessage(message);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initBleDataUI() {
        if (BleManager.getInstance(this).getmConnBleDevice() != null) {
            if (MyApplication.getApplication().getmProductModel() != null && Constans.DEVICE_TYPE_SMS.equals(MyApplication.getApplication().getmProductModel().getDeviceType())) {
                Message message = new Message();
                message.what = 1;
                message.obj = Integer.valueOf(BleManager.getInstance(this).getmConnBleDevice().getRunMode() == 0 ? 160 : 161);
                this.mHandler.sendMessage(message);
                int startPosition = BleManager.getInstance(this).getmConnBleDevice().getStartPosition();
                Message message2 = new Message();
                message2.what = 7;
                message2.obj = Integer.valueOf(startPosition);
                this.mHandler.sendMessage(message2);
                int stopPosition = BleManager.getInstance(this).getmConnBleDevice().getStopPosition();
                Message message3 = new Message();
                message3.what = 8;
                if (stopPosition <= 0) {
                    stopPosition = this.start_stop_position_diff + startPosition;
                }
                message3.obj = Integer.valueOf(stopPosition);
                this.mHandler.sendMessage(message3);
                Message message4 = new Message();
                message4.what = 9;
                message4.obj = Integer.valueOf(BleManager.getInstance(this).getmConnBleDevice().getSmoothness());
                this.mHandler.sendMessage(message4);
                Message message5 = new Message();
                message5.what = 10;
                message5.obj = Integer.valueOf(BleManager.getInstance(this).getmConnBleDevice().getPosition());
                this.mHandler.sendMessage(message5);
            }
            Message message6 = new Message();
            message6.what = 3;
            message6.obj = Integer.valueOf(BleManager.getInstance(this).getmConnBleDevice().getSpeed());
            this.mHandler.sendMessage(message6);
        }
    }

    private void initChatLayout() {
        boolean z = this.isShowMessage;
        ImageView imageView = this.mImgChat;
        if (z) {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_barrage_open));
            showChatLayout();
        } else {
            imageView.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_barrage_close));
            hideChatLayout();
        }
    }

    private void initModeDialog() {
        new XPopup.Builder(this).hasShadowBg(true).asCustom(this.mModePagerDialog).show();
    }

    private void initPreviewFragment(int i, String str, int i2, int i3) {
        if (i < 0) {
            closePreviewFragment();
        }
        this.mViewPreview.setVisibility(0);
        this.mPreviewFragment = new PreviewFragment2(i, str, i2, this.mSafeMax, this.mSafeVibrateMax, i3 == 0);
        if (MyApplication.getApplication().getmUserModel() != null && this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
            this.mPreviewFragment.setAllowClose(true);
            this.mPreviewFragment.setSeekBarClickable(true);
        }
        this.mPreviewFragment.setOnSpeedChangeListener(new PreviewFragment2.OnSpeedChangeListener() { // from class: com.hismith.hismithapp.LineRoomActivity.10
            @Override // com.hismith.hismithapp.fragment.PreviewFragment2.OnSpeedChangeListener
            public void onSpeedChange(int i4) {
                LineRoomActivity.this.setFloatSpeed(i4);
            }

            @Override // com.hismith.hismithapp.fragment.PreviewFragment2.OnSpeedChangeListener
            public void onVibrateChange(int i4) {
                LineRoomActivity.this.setFloatVibrate(i4);
            }
        });
        this.mPreviewFragment.setOnCloseClickListener(new PreviewFragment2.OnCloseClickListener() { // from class: com.hismith.hismithapp.LineRoomActivity.11
            @Override // com.hismith.hismithapp.fragment.PreviewFragment2.OnCloseClickListener
            public void onCloseClick(int i4) {
                LogUtil.e("onCloseClick()");
                LineRoomActivity.this.closePreviewFragment();
                final int i5 = LineRoomActivity.this.lastSpeed;
                if (i5 > LineRoomActivity.this.mSafeMax) {
                    i5 = LineRoomActivity.this.mSafeMax;
                }
                LineRoomActivity.this.setFloatSpeed(i5);
                if (ProductUtil.isSMSType() && i4 == 1) {
                    BleManager.getInstance(LineRoomActivity.this).writeRunAuto();
                    LineRoomActivity.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.LineRoomActivity.11.1
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(LineRoomActivity.this).writeSpeed(i5);
                        }
                    }, 25L);
                } else {
                    BleManager.getInstance(LineRoomActivity.this).writeSpeed(i5);
                }
                final int i6 = LineRoomActivity.this.lastVibrate;
                if (i6 > LineRoomActivity.this.mSafeVibrateMax) {
                    i6 = LineRoomActivity.this.mSafeVibrateMax;
                }
                if (ProductUtil.isSMVType()) {
                    LineRoomActivity.this.setFloatVibrate(i6);
                    LineRoomActivity.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.LineRoomActivity.11.2
                        @Override // java.lang.Runnable
                        public void run() {
                            BleManager.getInstance(LineRoomActivity.this).writeVibrate(i6);
                        }
                    }, 50L);
                }
                if (MyApplication.getApplication().getmUserModel() == null || !LineRoomActivity.this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
                    return;
                }
                LineRoomActivity.this.mPlayView.setTouch(true);
                LineRoomActivity.this.mPlayViewVibrate.setTouch(true);
                SocketManager.getInstance().setSpeed(i5);
                if (ProductUtil.isSMVType()) {
                    SocketManager.getInstance().setVibrate(i6);
                }
            }
        });
        getSupportFragmentManager().beginTransaction().replace(R.id.line_room_fragment_preview, this.mPreviewFragment, this.FRAGMENT_PREVIEW_TAG).commitAllowingStateLoss();
    }

    private void initProductData() {
        ProductModel productModel = MyApplication.getApplication().getmProductModel();
        if (productModel != null) {
            PlayView_Four playView_Four = this.mPlayView;
            if (playView_Four != null) {
                playView_Four.setmScreenSpeed(productModel.getSlide());
            }
            PlayView_Four playView_Four2 = this.mPlayViewVibrate;
            if (playView_Four2 != null) {
                playView_Four2.setmScreenSpeed(productModel.getSlide());
            }
            if (Constans.DEVICE_TYPE_SMS.equals(productModel.getDeviceType())) {
                ProgressBar progressBar = (ProgressBar) findViewById(R.id.progress_speed);
                this.mProgressSpeed = progressBar;
                this.mPlayView.setProgressBar(progressBar);
                this.mTxtSpeed = (TextView) findViewById(R.id.txt_speed);
                this.mLayoutSMSInfo.setVisibility(0);
                this.layout_sms_touch_position.setVisibility(0);
                this.mLayoutSpeedInfo.setVisibility(8);
                this.mLayoutVibrateInfo.setVisibility(8);
                this.mPlayViewVibrate.setVisibility(8);
                this.mViewLine.setVisibility(8);
                return;
            }
            if (Constans.DEVICE_TYPE_SMV.equals(productModel.getDeviceType())) {
                ProgressBar progressBar2 = (ProgressBar) findViewById(R.id.line_room_progress_speed);
                this.mProgressSpeed = progressBar2;
                this.mPlayView.setProgressBar(progressBar2);
                this.mTxtSpeed = (TextView) findViewById(R.id.line_room_txt_speed);
                this.mLayoutSMSInfo.setVisibility(8);
                this.layout_sms_touch_position.setVisibility(8);
                this.mLayoutSpeedInfo.setVisibility(0);
                this.mLayoutVibrateInfo.setVisibility(0);
                this.mPlayViewVibrate.setVisibility(0);
                this.mViewLine.setVisibility(0);
                return;
            }
            ProgressBar progressBar3 = (ProgressBar) findViewById(R.id.line_room_progress_speed);
            this.mProgressSpeed = progressBar3;
            this.mPlayView.setProgressBar(progressBar3);
            this.mTxtSpeed = (TextView) findViewById(R.id.line_room_txt_speed);
            this.mLayoutSMSInfo.setVisibility(8);
            this.layout_sms_touch_position.setVisibility(8);
            this.mLayoutSpeedInfo.setVisibility(0);
            this.mLayoutVibrateInfo.setVisibility(8);
            this.mPlayViewVibrate.setVisibility(8);
            this.mViewLine.setVisibility(8);
        }
    }

    private void initRecordPermission() {
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.LineRoomActivity$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                LineRoomActivity.this.m562x21c470aa((Permission) obj);
            }
        });
        if (this.mCompositeDisposable == null || this.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.mCompositeDisposable.add(subscribe);
    }

    private void initRightHallUserDialog() {
        this.rightHallUserDialog = new UserDrawerPopupDialog(this, "inviteTeammates", 1, new LineUserAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.LineRoomActivity.16
            @Override // com.hismith.hismithapp.adapter.LineUserAdapter.OnItemClickListener
            public void onItemClick(int i) {
                LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                LineRoomActivity lineRoomActivity2 = LineRoomActivity.this;
                lineRoomActivity.hallUserInfoDrawerPopupDialog = new UserInfoDrawerPopupDialog(lineRoomActivity2, (SocketUserModel) lineRoomActivity2.mHallUserList.get(i), 1, LineRoomActivity.this.captainId);
                new XPopup.Builder(LineRoomActivity.this).asCustom(LineRoomActivity.this.hallUserInfoDrawerPopupDialog).show();
            }
        });
        SocketManager.getInstance().hallUser();
        new XPopup.Builder(this).asCustom(this.rightHallUserDialog).show();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initRightRoomUserDialog() {
        this.rightRoomUserDialog = new UserDrawerPopupDialog(this, "teammate", 0, new LineUserAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.LineRoomActivity.15
            @Override // com.hismith.hismithapp.adapter.LineUserAdapter.OnItemClickListener
            public void onItemClick(int i) {
                LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                LineRoomActivity lineRoomActivity2 = LineRoomActivity.this;
                lineRoomActivity.roomUserInfoDrawerPopupDialog = new UserInfoDrawerPopupDialog(lineRoomActivity2, (SocketUserModel) lineRoomActivity2.mRoomUserList.get(i), 0, LineRoomActivity.this.captainId);
                new XPopup.Builder(LineRoomActivity.this).asCustom(LineRoomActivity.this.roomUserInfoDrawerPopupDialog).show();
            }
        });
        SocketManager.getInstance().roomUser();
        new XPopup.Builder(this).asCustom(this.rightRoomUserDialog).show();
    }

    private void initRoomInfo(LineRoomInfoModel lineRoomInfoModel) {
        PreviewFragment2 previewFragment2;
        this.mTxtRoomId.setText("ID: " + SocketManager.getInstance().getRoomId());
        Glide.with((FragmentActivity) this).load(lineRoomInfoModel.getUserHeadIcon()).placeholder(R.mipmap.ic_personal_head).into((RequestBuilder) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.LineRoomActivity.13
            public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                LineRoomActivity.this.mImgOwnerHead.setImageDrawable(drawable);
            }

            @Override // com.bumptech.glide.request.target.Target
            public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
            }
        });
        this.mTxtOwnerName.setText(lineRoomInfoModel.getUserName());
        this.mTxtUserCount.setText(lineRoomInfoModel.getUserCount() + "");
        int roomIsPublic = SocketManager.getInstance().getRoomIsPublic();
        ImageView imageView = this.mImgRoomPublic;
        int i = 0;
        if (roomIsPublic == 1) {
            imageView.setVisibility(8);
        } else {
            imageView.setVisibility(0);
        }
        if (SocketManager.getInstance().getRoomMessageModel() != null) {
            this.mMessages.add(SocketManager.getInstance().getRoomMessageModel());
            SocketManager.getInstance().setRoomMessageModel(null);
            this.mMessageAdapter.notifyDataSetChanged();
            if (this.isRoomOrWorld == 0) {
                this.mRecyclerMessage.smoothScrollToPosition(this.mMessages.size() - 1);
            }
        }
        this.mUsers.clear();
        this.mPlayView.setTouch(false);
        this.mPlayViewVibrate.setTouch(false);
        String str = "";
        for (LineRoomInfoModel.RoomInfoUserModel roomInfoUserModel : lineRoomInfoModel.getUser()) {
            if (roomInfoUserModel.getCaptain() == 1) {
                str = roomInfoUserModel.getUserId();
            }
            this.mUsers.add(roomInfoUserModel);
        }
        if (TextUtils.isEmpty(str)) {
            str = lineRoomInfoModel.getUserId();
        }
        if (!this.captainId.equals(str)) {
            this.captainId = str;
            if (this.isModeing && MyApplication.getApplication().getmUserModel() != null && this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "") && (previewFragment2 = this.mPreviewFragment) != null) {
                previewFragment2.setAllowClose(true);
                this.mPreviewFragment.setSeekBarClickable(true);
            }
        }
        this.mProgressSpeed.setSecondaryProgress(0);
        this.mProgressVibrate.setSecondaryProgress(0);
        if (MyApplication.getApplication().getmUserModel() == null || !this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
            this.mImgMode.setVisibility(8);
            this.img_run_mode.setEnabled(false);
            if (lineRoomInfoModel.getPlay() != null) {
                LineRoomInfoModel.Play play = lineRoomInfoModel.getPlay();
                int speed = play.getSpeed();
                if (speed > 100) {
                    speed = 100;
                } else if (speed < 0) {
                    speed = 0;
                }
                this.lastSpeed = speed;
                int i2 = this.mSafeMax;
                if (speed <= i2) {
                    i2 = speed;
                }
                this.mProgressSpeed.setSecondaryProgress(speed);
                int vspeed = play.getVspeed();
                if (vspeed > 100) {
                    i = 100;
                } else if (vspeed >= 0) {
                    i = vspeed;
                }
                this.lastVibrate = i;
                final int i3 = this.mSafeVibrateMax;
                if (i <= i3) {
                    i3 = i;
                }
                this.mProgressVibrate.setSecondaryProgress(i);
                if (TextUtils.isEmpty(play.getModePoints())) {
                    BleManager.getInstance(this).writeSpeed(i2);
                    if (i3 > 0) {
                        this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.LineRoomActivity.14
                            @Override // java.lang.Runnable
                            public void run() {
                                BleManager.getInstance(LineRoomActivity.this).writeVibrate(i3);
                            }
                        }, 100L);
                    }
                } else {
                    this.isModeing = true;
                    this.mModeingId = play.getMode();
                    initPreviewFragment(play.getMode(), play.getModePoints(), (int) (play.getPlayTime() * 1000.0f), play.getModeStop());
                }
            }
        } else {
            this.isSync = true;
            this.mWebViewRocket.setVisibility(8);
            this.mWebView.setVisibility(8);
            this.mImgStart.setVisibility(8);
            this.img_run_mode.setEnabled(true);
            BleManager.getInstance(this).setWriteSpeed(true);
            this.mImgMode.setVisibility(0);
            if (!this.isModeing) {
                this.mPlayView.setTouch(true);
                this.mPlayViewVibrate.setTouch(true);
            }
            if (MyApplication.getApplication().getmProductModel() != null && Constans.DEVICE_TYPE_SMS.equals(MyApplication.getApplication().getmProductModel().getDeviceType()) && BleManager.getInstance(this).getmConnBleDevice() != null) {
                if (BleManager.getInstance(this).getmConnBleDevice().getRunMode() == 1) {
                    BleManager.getInstance(this).writeRunManual();
                } else {
                    BleManager.getInstance(this).writeRunAuto();
                }
            }
        }
        this.mHeadAdapter.notifyDataSetChanged();
    }

    private void initView() {
        View findViewById = findViewById(R.id.view_network_error);
        this.mViewNetworkError = findViewById;
        this.mImgNetworkError = (ImageView) findViewById.findViewById(R.id.img_network_error);
        this.mTxtNetworkError = (TextView) this.mViewNetworkError.findViewById(R.id.txt_network_error);
        this.mLayoutBg = (ImageView) findViewById(R.id.line_room_layout_bg);
        this.mPlayView = (PlayView_Four) findViewById(R.id.line_room_playView);
        this.mPlayViewVibrate = (PlayView_Four) findViewById(R.id.line_room_playView_vibrate);
        this.mViewLine = findViewById(R.id.line_room_line);
        this.mImgOwnerHead = (CircleImageView) findViewById(R.id.line_room_img_owner_head);
        this.mTxtOwnerName = (TextView) findViewById(R.id.line_room_txt_owner_name);
        this.mLayoutUserCount = (LinearLayout) findViewById(R.id.line_room_layout_user_count);
        this.mTxtUserCount = (TextView) findViewById(R.id.line_room_txt_user_count);
        this.mTxtRoomId = (TextView) findViewById(R.id.line_room_txt_room_id);
        this.mLayoutMaxSpeed = (ViewGroup) findViewById(R.id.line_room_layout_max_speed);
        this.mTxtMaxSpeed = (TextView) findViewById(R.id.line_room_txt_max_speed);
        this.mTxtSpeed = (TextView) findViewById(R.id.line_room_txt_speed);
        this.mImgRoomPublic = (ImageView) findViewById(R.id.line_room_img_room_public);
        this.mImgUserInvite = (ImageView) findViewById(R.id.line_room_img_user_invite);
        this.mImgSound = (ImageView) findViewById(R.id.line_room_img_sound);
        this.mImgVoice = (ImageView) findViewById(R.id.line_room_img_voice);
        this.mImgChat = (ImageView) findViewById(R.id.line_room_img_chat);
        this.mImgExit = (ImageView) findViewById(R.id.line_room_img_exit);
        this.mRecyclerUser = (RecyclerView) findViewById(R.id.line_room_recycler_user);
        this.mRecyclerMessage = (RecyclerView) findViewById(R.id.line_room_recycler_message);
        this.mMessageResizeHandle = findViewById(R.id.line_room_message_resize_handle);
        this.mProgressSpeed = (ProgressBar) findViewById(R.id.line_room_progress_speed);
        this.mEditMessage = (EditText) findViewById(R.id.line_room_edit_message);
        this.mImgMode = (ImageView) findViewById(R.id.line_room_img_mode);
        this.mViewPreview = (FrameLayout) findViewById(R.id.line_room_fragment_preview);
        this.mImgStart = (ImageView) findViewById(R.id.line_room_img_start);
        this.mImgCrown = (ImageView) findViewById(R.id.line_room_img_user_crown);
        this.mLayoutSpeedInfo = (RelativeLayout) findViewById(R.id.line_room_layout_speed_info);
        this.mLayoutVibrateInfo = (RelativeLayout) findViewById(R.id.line_room_layout_vibrate_info);
        this.mProgressVibrate = (VerticalProgressBar) findViewById(R.id.line_room_progress_vibrate);
        this.mLayoutMaxVibrate = (ViewGroup) findViewById(R.id.line_room_layout_max_vibrate);
        this.mTxtMaxVibrate = (TextView) findViewById(R.id.line_room_txt_max_vibrate);
        this.mTxtVibrate = (TextView) findViewById(R.id.line_room_txt_vibrate);
        this.mLayoutSMSInfo = (LinearLayout) findViewById(R.id.line_room_layout_sms_info);
        this.img_run_mode = (ImageView) findViewById(R.id.img_run_mode);
        this.txt_speed_label = (TextView) findViewById(R.id.txt_speed_label);
        this.txt_speed = (TextView) findViewById(R.id.txt_speed);
        this.progress_speed = (ProgressBar) findViewById(R.id.progress_speed);
        this.layout_speed_max = (ViewGroup) findViewById(R.id.layout_speed_max);
        this.txt_speed_max = (TextView) findViewById(R.id.txt_speed_max);
        this.txt_position_label = (TextView) findViewById(R.id.txt_position_label);
        this.txt_position = (TextView) findViewById(R.id.txt_position);
        this.progress_position = (ProgressBar) findViewById(R.id.progress_position);
        this.line_max_start = (Guideline) findViewById(R.id.line_max_start);
        this.line_min_start = (Guideline) findViewById(R.id.line_min_start);
        this.layout_position_max = (ViewGroup) findViewById(R.id.layout_position_max);
        this.layout_position_min = (ViewGroup) findViewById(R.id.layout_position_min);
        this.txt_position_max = (TextView) findViewById(R.id.txt_position_max);
        this.txt_position_min = (TextView) findViewById(R.id.txt_position_min);
        this.txt_smooth_label = (TextView) findViewById(R.id.txt_smooth_label);
        this.txt_smooth = (TextView) findViewById(R.id.txt_smooth);
        this.seek_smooth = (CustomSeekbar) findViewById(R.id.seek_smooth);
        this.layout_sms_touch_position = (ConstraintLayout) findViewById(R.id.layout_sms_touch_position);
        this.layout_touch_position = (ConstraintLayout) findViewById(R.id.layout_touch_position);
        this.img_position = (ImageView) findViewById(R.id.img_position);
        LanguageUtil.setText(this, this.txt_speed_label, "speedTitle");
        this.txt_position_label.setText(LanguageUtil.getLanguageByKey(this, "position") + ":");
        this.txt_smooth_label.setText(LanguageUtil.getLanguageByKey(this, "smoothness") + ":");
        this.mWebView = (CustomWebView) findViewById(R.id.web_view);
        this.mWebViewRocket = (CustomWebView) findViewById(R.id.web_view_rocket);
        if (this.mLayoutBg != null) {
            Glide.with((FragmentActivity) this).load(Integer.valueOf(R.mipmap.bg_iplay)).into(this.mLayoutBg);
        }
        this.mImgMode.setVisibility(8);
    }

    private void initWebView() {
        ViewGroup.LayoutParams layoutParams = this.mWebView.getLayoutParams();
        layoutParams.width = (int) (MeasureUtil.getScreenWidth(this) * 0.7f);
        this.mWebView.setLayoutParams(layoutParams);
        this.mWebView.getSettings().setCacheMode(1);
        this.mWebViewRocket.getSettings().setCacheMode(1);
        this.mWebView.setBackgroundColor(0);
        this.mWebView.setWebViewClient(new GameWebViewClient());
        this.mWebView.setInitialScale(100);
        this.mWebView.getSettings().setJavaScriptEnabled(true);
        this.mWebViewRocket.setBackgroundColor(0);
        this.mWebViewRocket.setWebViewClient(new GameWebViewClient());
        this.mWebViewRocket.setInitialScale(100);
        this.mWebViewRocket.getSettings().setJavaScriptEnabled(true);
        if (TextUtils.isEmpty("file:///android_asset/yanwu.html")) {
            LogUtil.e("webView uri:NULL");
        } else {
            LogUtil.e("webView uri:file:///android_asset/yanwu.html");
            this.mWebView.loadUrl("file:///android_asset/yanwu.html");
        }
        if (TextUtils.isEmpty("file:///android_asset/huojian.html")) {
            LogUtil.e("webView uri:NULL");
        } else {
            LogUtil.e("webView uri:file:///android_asset/huojian.html");
            this.mWebViewRocket.loadUrl("file:///android_asset/huojian.html");
        }
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void launchActivity(Context context) {
        Intent intent = new Intent("android.intent.action.MAIN");
        intent.addCategory("android.intent.category.LAUNCHER");
        intent.setFlags(270532608);
        intent.setComponent(new ComponentName(getPackageName(), "com.hismith.hismithapp.LineRoomActivity"));
        MyApplication.getApplication().startActivity(intent);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void manualTouchPosition() {
        int position = BleManager.getInstance(this).getmConnBleDevice().getPosition();
        float stopPosition = BleManager.getInstance(this).getmConnBleDevice().getStopPosition() - BleManager.getInstance(this).getmConnBleDevice().getStartPosition();
        final float x = this.img_position.getX() + (this.img_position.getWidth() / 2.0f);
        final float height = (((this.layout_touch_position.getHeight() - this.img_position.getHeight()) / stopPosition) * (stopPosition - position)) + (this.img_position.getHeight() / 2.0f);
        this.img_position.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.LineRoomActivity.30
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                LineRoomActivity.this.img_position.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                lineRoomActivity.changeUIPosition(lineRoomActivity.layout_touch_position, LineRoomActivity.this.img_position, x, height);
            }
        });
    }

    private void openMic() {
        LogUtil.e("打开麦克风");
        if (!TextUtils.isEmpty(this.isRoomVoice) && "1".equals(this.isRoomVoice) && AudioSocketManager.getInstance().openVoice()) {
            this.isCloseMic = false;
            this.mImgVoice.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_line_voice));
            LineAudioManager.getInstance().startAudio();
        }
    }

    private void pauseBle() {
        BleManager.getInstance(this).writeShutdown();
    }

    private void recoverBle() {
        int i = this.lastSpeed;
        int i2 = this.mSafeMax;
        if (i > i2) {
            i = i2;
        }
        BleManager.getInstance(this).writeSpeed(i);
        this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.LineRoomActivity.40
            @Override // java.lang.Runnable
            public void run() {
                int i3 = LineRoomActivity.this.lastVibrate;
                if (i3 > LineRoomActivity.this.mSafeVibrateMax) {
                    i3 = LineRoomActivity.this.mSafeVibrateMax;
                }
                BleManager.getInstance(LineRoomActivity.this).writeVibrate(i3);
            }
        }, 100L);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshEndBitUI(final int i) {
        this.mPositionEnd = i;
        this.txt_position_max.setText(this.mPositionEnd + "");
        final int width = this.progress_position.getWidth();
        if (width == 0) {
            this.progress_position.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.LineRoomActivity.31
                @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
                public void onGlobalLayout() {
                    LineRoomActivity.this.progress_position.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                    LogUtil.e("refreshEndBitUI sumWidth:" + width);
                    LogUtil.e("refreshEndBitUI endBit:" + i);
                    LineRoomActivity.this.refreshEndMargins(i);
                }
            });
        } else {
            refreshEndMargins(i);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshEndMargins(int i) {
        double width = this.progress_position.getWidth();
        int i2 = (int) (width - ((width / 100.0d) * i));
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_position_max.getLayoutParams();
        layoutParams.setMargins(0, 0, i2, 0);
        this.layout_position_max.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshPositionUI(int i) {
        this.mPosition = i;
        this.txt_position.setText(i + "");
        this.progress_position.setProgress(i);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSafeSMSUI(int i) {
        this.txt_speed_max.setText(i + "");
        double width = this.progress_speed.getWidth();
        int i2 = (int) (width - ((width / 100.0d) * i));
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_speed_max.getLayoutParams();
        layoutParams.setMargins(0, 0, i2, 0);
        this.layout_speed_max.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSafeUI(int i) {
        this.mTxtMaxSpeed.setText(i + "");
        double height = this.mProgressSpeed.getHeight();
        int i2 = (int) (height - ((height / 100.0d) * i));
        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) this.mLayoutMaxSpeed.getLayoutParams();
        layoutParams.setMargins(0, i2, 0, 0);
        this.mLayoutMaxSpeed.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSmoothUI(int i) {
        this.txt_smooth.setText(i + "");
        ValueAnimator ofInt = ValueAnimator.ofInt(this.seek_smooth.getProgress(), i * 100);
        ofInt.setInterpolator(new LinearInterpolator());
        ofInt.setDuration(300L);
        ofInt.setRepeatCount(0);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.LineRoomActivity.33
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                LogUtil.e("refreshSmoothUI:" + ((Integer) valueAnimator.getAnimatedValue()));
                LineRoomActivity.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
            }
        });
        ofInt.start();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshStartBitUI(final int i) {
        this.mPositionStart = i;
        this.txt_position_min.setText(this.mPositionStart + "");
        final int width = this.progress_position.getWidth();
        if (width == 0) {
            this.progress_position.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.LineRoomActivity.32
                @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
                public void onGlobalLayout() {
                    LineRoomActivity.this.progress_position.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                    LogUtil.e("refreshEndBitUI sumWidth:" + width);
                    LogUtil.e("refreshEndBitUI startBit:" + i);
                    LineRoomActivity.this.refreshStartMargins(i);
                }
            });
        } else {
            refreshStartMargins(i);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshStartMargins(int i) {
        ConstraintLayout.LayoutParams layoutParams = (ConstraintLayout.LayoutParams) this.layout_position_min.getLayoutParams();
        layoutParams.setMargins((int) ((this.progress_position.getWidth() / 100.0d) * i), 0, 0, 0);
        this.layout_position_min.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshVibrateSafeUI(int i) {
        this.mTxtMaxVibrate.setText(i + "");
        double height = this.mProgressVibrate.getHeight();
        int i2 = (int) (height - ((height / 100.0d) * i));
        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) this.mLayoutMaxVibrate.getLayoutParams();
        layoutParams.setMargins(0, i2, 0, 0);
        this.mLayoutMaxVibrate.setLayoutParams(layoutParams);
    }

    private void registerNetReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction("android.net.conn.CONNECTIVITY_CHANGE");
        NetBroadcastReceiver netBroadcastReceiver = new NetBroadcastReceiver();
        this.mNetBroadcastReceiver = netBroadcastReceiver;
        netBroadcastReceiver.setNetChangeListener(this);
        AppUtil.registerReceiver(this, this.mNetBroadcastReceiver, intentFilter);
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void resizeMessageLayout(int i, int i2) {
        int dp2px = MeasureUtil.dp2px(this, 200.0f);
        int dp2px2 = MeasureUtil.dp2px(this, 59.0f);
        int screenWidth = MeasureUtil.getScreenWidth(this);
        int screenHeight = MeasureUtil.getScreenHeight(this);
        int left = this.mRecyclerMessage.getLeft();
        int min = Math.min((int) (screenHeight * MESSAGE_MAX_SIZE_HEIGHT), this.mRecyclerMessage.getBottom());
        int max = Math.max(dp2px, screenWidth - (left * 2));
        int max2 = Math.max(dp2px2, min);
        int max3 = Math.max(dp2px, Math.min(i, max));
        int max4 = Math.max(dp2px2, Math.min(i2, max2));
        ViewGroup.LayoutParams layoutParams = this.mRecyclerMessage.getLayoutParams();
        layoutParams.width = max3;
        layoutParams.height = max4;
        this.mRecyclerMessage.setLayoutParams(layoutParams);
        this.mRecyclerMessageHeight = max4;
        if (max4 > MeasureUtil.dp2px(this, 59.0f)) {
            this.isShowMessage = true;
            this.mImgChat.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_barrage_open));
        } else {
            this.isShowMessage = false;
            this.mImgChat.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_barrage_close));
            this.mRecyclerMessage.scrollToPosition(this.mMessages.size() - 1);
        }
    }

    private void rocketAnimation() {
        final int i = (int) (-(this.mWebViewRocket.getY() + this.mWebViewRocket.getHeight()));
        final int top = this.mWebViewRocket.getTop();
        final int bottom = this.mWebViewRocket.getBottom();
        ValueAnimator ofInt = ValueAnimator.ofInt(0, i);
        ofInt.setInterpolator(new LinearInterpolator());
        ofInt.setDuration(1000L);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.LineRoomActivity.17
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                int intValue = ((Integer) valueAnimator.getAnimatedValue()).intValue();
                float f = 1.0f - ((intValue * 1.0f) / i);
                LogUtil.e("onAnimationUpdate:" + f);
                LineRoomActivity.this.mWebViewRocket.layout(LineRoomActivity.this.mWebViewRocket.getLeft(), top + intValue, LineRoomActivity.this.mWebViewRocket.getRight(), bottom + intValue);
                LineRoomActivity.this.mWebView.setAlpha(f);
            }
        });
        ofInt.addListener(new AnimatorListenerAdapter() { // from class: com.hismith.hismithapp.LineRoomActivity.18
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator, boolean z) {
                if (LineRoomActivity.this.mPreviewFragment != null) {
                    LineRoomActivity.this.mPreviewFragment.attachedToStartRun();
                }
                LineRoomActivity.this.isSync = true;
                BleManager.getInstance(LineRoomActivity.this).setWriteSpeed(true);
                if (!LineRoomActivity.this.isModeing) {
                    if (MyApplication.getApplication().getmProductModel() == null || !Constans.DEVICE_TYPE_SMS.equals(MyApplication.getApplication().getmProductModel().getDeviceType()) || BleManager.getInstance(LineRoomActivity.this).getmConnBleDevice() == null || BleManager.getInstance(LineRoomActivity.this).getmConnBleDevice().getRunMode() != 1) {
                        int i2 = LineRoomActivity.this.lastSpeed;
                        if (i2 > LineRoomActivity.this.mSafeMax) {
                            i2 = LineRoomActivity.this.mSafeMax;
                        }
                        BleManager.getInstance(LineRoomActivity.this).writeSpeed(i2);
                        LineRoomActivity.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.LineRoomActivity.18.1
                            @Override // java.lang.Runnable
                            public void run() {
                                int i3 = LineRoomActivity.this.lastVibrate;
                                if (i3 > LineRoomActivity.this.mSafeVibrateMax) {
                                    i3 = LineRoomActivity.this.mSafeVibrateMax;
                                }
                                BleManager.getInstance(LineRoomActivity.this).writeVibrate(i3);
                            }
                        }, 100L);
                    } else {
                        BleManager.getInstance(LineRoomActivity.this).writePosition(LineRoomActivity.this.recePosition);
                    }
                }
                LineRoomActivity.this.mWebViewRocket.clearAnimation();
                LineRoomActivity.this.mWebViewRocket.setVisibility(8);
                LineRoomActivity.this.mWebView.setVisibility(8);
                LineRoomActivity.this.mImgStart.setVisibility(8);
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator, boolean z) {
                if (LineRoomActivity.this.mPreviewFragment != null) {
                    LineRoomActivity.this.mPreviewFragment.detachedToStopRun();
                }
                LineRoomActivity.this.mWebViewRocket.setVisibility(0);
                LineRoomActivity.this.mWebView.setVisibility(0);
                LineRoomActivity.this.mImgStart.setVisibility(4);
            }
        });
        ofInt.start();
    }

    private void setFloatPosition(int i) {
        FloatBase floatBase = this.mFloatBase;
        if (floatBase != null) {
            floatBase.refreshFloatPosition(i);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setFloatSpeed(int i) {
        FloatBase floatBase = this.mFloatBase;
        if (floatBase != null) {
            floatBase.refreshFloatSpeedUI(i);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void setFloatVibrate(int i) {
        FloatBase floatBase = this.mFloatBase;
        if (floatBase != null) {
            floatBase.refreshFloatVibrateUI(i);
        }
    }

    private void setListener() {
        SocketManager.getInstance().setOnSocketNotifyListener(this);
        AudioSocketManager.getInstance().setOnAudioSocketNotifyListener(this);
        this.mViewNetworkError.setOnClickListener(this);
        this.mImgOwnerHead.setOnClickListener(this);
        this.mImgSound.setOnClickListener(this);
        this.mImgVoice.setOnClickListener(this);
        this.mImgChat.setOnClickListener(this);
        this.mImgExit.setOnClickListener(this);
        this.mLayoutMaxSpeed.setOnTouchListener(this);
        this.mLayoutMaxVibrate.setOnTouchListener(this.vibrateSafeTouch);
        this.mLayoutUserCount.setOnClickListener(this);
        this.mImgUserInvite.setOnClickListener(this);
        this.mImgMode.setOnClickListener(this);
        this.mImgStart.setOnClickListener(this);
        this.mMessageResizeHandle.setOnTouchListener(this.messageResizeTouch);
        this.img_run_mode.setOnClickListener(this);
        this.layout_speed_max.setOnTouchListener(this.safeOnTouch);
        this.layout_position_max.setOnTouchListener(this.endOnTouch);
        this.layout_position_min.setOnTouchListener(this.startOnTouch);
        this.layout_touch_position.setOnTouchListener(this.positionOnTouch);
        this.mPlayView.setOnLineChangeListener(this);
        this.mPlayViewVibrate.setOnLineChangeListener(new PlayView_Four.OnLineChangeListener() { // from class: com.hismith.hismithapp.LineRoomActivity.3
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnLineChangeListener
            public void onLineChange(int i, JsonObject jsonObject) {
                if (i > 100) {
                    i = 100;
                } else if (i < 0) {
                    i = 0;
                }
                if (i > LineRoomActivity.this.mSafeVibrateMax) {
                    i = LineRoomActivity.this.mSafeVibrateMax;
                }
                if (i != LineRoomActivity.this.mProgressVibrate.getProgress() || i == 0) {
                    LineRoomActivity.this.lastVibrate = i;
                    BleManager.getInstance(LineRoomActivity.this).writeVibrate(i);
                    SocketManager.getInstance().setVibrate(i, jsonObject);
                }
                LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                SoftKeyboardUtil.hideSoftKeyboard(lineRoomActivity, lineRoomActivity.mPlayViewVibrate);
                if (LineRoomActivity.this.mEditMessage.hasFocus()) {
                    LineRoomActivity.this.mEditMessage.clearFocus();
                }
            }
        });
        this.mEditMessage.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.LineRoomActivity.4
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 4) {
                    return false;
                }
                String obj = textView.getText().toString();
                if (obj.length() > 128) {
                    return true;
                }
                textView.setText("");
                if (!TextUtils.isEmpty(obj)) {
                    if (LineRoomActivity.this.isRoomOrWorld == 0) {
                        SocketManager.getInstance().roomMessage(obj);
                    } else if (LineRoomActivity.this.isRoomOrWorld == 1) {
                        SocketManager.getInstance().worldMessage(obj);
                    }
                }
                SoftKeyboardUtil.hideSoftKeyboard(LineRoomActivity.this, textView);
                return true;
            }
        });
        this.mHeadAdapter.setOnItemClickListener(new CircleImgAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.LineRoomActivity.5
            @Override // com.hismith.hismithapp.adapter.CircleImgAdapter.OnItemClickListener
            public void onItemClick(int i) {
                LineRoomActivity.this.initRightRoomUserDialog();
            }
        });
        this.seek_smooth.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.LineRoomActivity.6
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
                LogUtil.e("onProgressChanged:" + i);
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
                LogUtil.e("onStartTrackingTouch:" + LineRoomActivity.this.seek_smooth.getProgress());
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
                int i;
                LogUtil.e("onStopTrackingTouch:" + LineRoomActivity.this.seek_smooth.getProgress());
                int progress = LineRoomActivity.this.seek_smooth.getProgress();
                int i2 = progress % 100;
                if (i2 < 50) {
                    i = progress - i2;
                    LogUtil.e("newProgress:" + i);
                } else {
                    i = (progress - i2) + 100;
                    LogUtil.e("newProgress:" + i);
                }
                int i3 = i / 100;
                LineRoomActivity.this.txt_smooth.setText(i3 + "");
                BleManager.getInstance(LineRoomActivity.this).writeSmooth(i3);
                ValueAnimator ofInt = ValueAnimator.ofInt(progress, i);
                ofInt.setInterpolator(new LinearInterpolator());
                ofInt.setDuration(300L);
                ofInt.setRepeatCount(0);
                ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.LineRoomActivity.6.1
                    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                    public void onAnimationUpdate(ValueAnimator valueAnimator) {
                        LineRoomActivity.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
                    }
                });
                ofInt.start();
            }
        });
        this.mLayoutMaxSpeed.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.LineRoomActivity.7
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                LineRoomActivity.this.mLayoutMaxSpeed.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                lineRoomActivity.refreshSafeUI(lineRoomActivity.mSafeMax);
            }
        });
        this.layout_speed_max.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.LineRoomActivity.8
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                LineRoomActivity.this.layout_speed_max.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                lineRoomActivity.refreshSafeSMSUI(lineRoomActivity.mSafeMax);
            }
        });
        this.mLayoutMaxVibrate.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() { // from class: com.hismith.hismithapp.LineRoomActivity.9
            @Override // android.view.ViewTreeObserver.OnGlobalLayoutListener
            public void onGlobalLayout() {
                LineRoomActivity.this.mLayoutMaxVibrate.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                lineRoomActivity.refreshVibrateSafeUI(lineRoomActivity.mSafeVibrateMax);
            }
        });
    }

    private void showChatLayout() {
        if (this.mRecyclerMessageHeight == MeasureUtil.dp2px(this, 59.0f)) {
            this.mRecyclerMessageHeight = MeasureUtil.dp2px(this, 120.0f);
        }
        ValueAnimator ofInt = ValueAnimator.ofInt(this.mRecyclerMessage.getHeight(), this.mRecyclerMessageHeight);
        this.mMessageLayoutAnimator = ofInt;
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.LineRoomActivity.35
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                int intValue = ((Integer) valueAnimator.getAnimatedValue()).intValue();
                LogUtil.e("onAnimationUpdate h:" + intValue);
                LineRoomActivity.this.mRecyclerMessage.getLayoutParams().height = intValue;
                LineRoomActivity.this.mRecyclerMessage.requestLayout();
            }
        });
        this.mMessageLayoutAnimator.setDuration(this.animationDuration);
        this.mMessageLayoutAnimator.start();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void showExitDialog() {
        DialogUtil.showConfirmDialog(this, "exitRoom", null, null, "exitButton", new View.OnClickListener() { // from class: com.hismith.hismithapp.LineRoomActivity.19
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                LineRoomActivity.this.appManager.finishActivity(LineRoomActivity.class);
                LineRoomActivity.this.finish();
                DialogUtil.dismissDialog();
            }
        });
    }

    private void showFloatWindow() {
        if (CheckUtil.checkOp(this, 24) == 0) {
            FloatBase floatBase = this.mFloatBase;
            if (floatBase == null) {
                floatBase = FloatManager.getFloat(this, MyApplication.getApplication().getmProductModel() != null ? MyApplication.getApplication().getmProductModel().getDeviceType() : "", Constans.FLOAT_LINE_WINDOW, new View.OnClickListener() { // from class: com.hismith.hismithapp.LineRoomActivity.20
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view) {
                        LogUtil.e("BleManager 点击悬浮窗");
                        if (AppBackgroundUtil.isAppBackground()) {
                            return;
                        }
                        LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                        lineRoomActivity.launchActivity(lineRoomActivity);
                        LineRoomActivity.this.destroyFloatWindow();
                    }
                });
                this.mFloatBase = floatBase;
            }
            if (floatBase != null) {
                floatBase.show();
            }
        }
    }

    private void socketAudioFailure() {
        int i;
        LineAudioManager.getInstance().stopAudio();
        int i2 = this.mDisconnType;
        if ((i2 & 2) == 0) {
            this.mDisconnType = i2 + 2;
        }
        if (this.mLastNetStatus == 0 || (i = this.mAudioReconnCount) <= 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.LineRoomActivity.23
                @Override // java.lang.Runnable
                public void run() {
                    if (LineRoomActivity.this.mTxtNetworkError != null) {
                        LineRoomActivity.this.mViewNetworkError.setVisibility(0);
                        LineRoomActivity.this.mTxtNetworkError.setText("Audio " + LanguageUtil.getLanguageByKey(LineRoomActivity.this, "notNetwork"));
                    }
                }
            });
            return;
        }
        this.mAudioReconnCount = i - 1;
        if (!TextUtils.isEmpty(this.isRoomVoice) && "1".equals(this.isRoomVoice)) {
            AudioSocketManager.getInstance().connSocket(3, this.isCloseSpeaker);
        }
        if (this.mHandler.hasMessages(50)) {
            this.mHandler.removeMessages(50);
        }
        this.mHandler.sendEmptyMessageDelayed(50, this.mSocketFailureDelay);
    }

    private void socketFailure() {
        int i;
        int i2 = this.mDisconnType;
        if ((i2 & 1) == 0) {
            this.mDisconnType = i2 + 1;
        }
        if (this.mLastNetStatus == 0 || (i = this.mReconnCount) <= 0) {
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.LineRoomActivity.22
                @Override // java.lang.Runnable
                public void run() {
                    if (LineRoomActivity.this.mTxtNetworkError != null) {
                        LineRoomActivity.this.mViewNetworkError.setVisibility(0);
                        LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                        LanguageUtil.setText(lineRoomActivity, lineRoomActivity.mTxtNetworkError, "notNetwork");
                    }
                }
            });
            return;
        }
        this.mReconnCount = i - 1;
        SocketManager.getInstance().connSocket(2);
        if (this.mHandler.hasMessages(50)) {
            this.mHandler.removeMessages(50);
        }
        this.mHandler.sendEmptyMessageDelayed(50, this.mSocketFailureDelay);
    }

    private void unregisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.manager.AudioSocketManager.OnAudioSocketNotifyListener
    public void OnAudioNotify(String str) {
        try {
            JsonObject asJsonObject = new JsonParser().parse(str).getAsJsonObject();
            if (1 == (asJsonObject.get("success") == null ? 0 : asJsonObject.get("success").getAsInt())) {
                String str2 = "";
                String asString = asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION) == null ? "" : asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION).getAsString();
                if (Constans.roomVoice.equals(asString)) {
                    if (this.isCloseSpeaker) {
                        return;
                    }
                    String asString2 = asJsonObject.get("voiceData") == null ? "" : asJsonObject.get("voiceData").getAsString();
                    String asString3 = asJsonObject.get("userId") == null ? "" : asJsonObject.get("userId").getAsString();
                    if (TextUtils.isEmpty(asString2)) {
                        return;
                    }
                    LineAudioManager.getInstance().playAudio(asString3, asString2.replaceAll("''", ""));
                    return;
                }
                if ("openVoice".equals(asString) || "openSound".equals(asString) || "closeVoice".equals(asString) || "closeSound".equals(asString)) {
                    if (asJsonObject.get("voiceStatus") != null) {
                        str2 = asJsonObject.get("voiceStatus").getAsString();
                    }
                    if (asJsonObject.get("soundStatus") != null) {
                        asJsonObject.get("soundStatus").getAsString();
                    }
                    LineAudioManager.getInstance().setIsSocketMic(str2);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void OnNotify(String str) {
        try {
            final JsonObject asJsonObject = new JsonParser().parse(str).getAsJsonObject();
            if (1 == (asJsonObject.get("success") == null ? 0 : asJsonObject.get("success").getAsInt())) {
                PlayLineSocketModel playLineSocketModel = (PlayLineSocketModel) this.gson.fromJson(str, PlayLineSocketModel.class);
                String str2 = "";
                String asString = asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION) == null ? "" : asJsonObject.get(NativeProtocol.WEB_DIALOG_ACTION).getAsString();
                if ("onlinePlay".equals(asString)) {
                    if ((asJsonObject.get("userRoomId") == null ? 0 : asJsonObject.get("userRoomId").getAsInt()) < 1) {
                        this.appManager.finishActivity(LineRoomActivity.class);
                        finish();
                        return;
                    }
                    return;
                }
                if ("roomInfo".equals(asString)) {
                    LineRoomInfoModel lineRoomInfoModel = (LineRoomInfoModel) this.gson.fromJson(str, LineRoomInfoModel.class);
                    if (lineRoomInfoModel != null) {
                        initRoomInfo(lineRoomInfoModel);
                    }
                    UserInfoDrawerPopupDialog userInfoDrawerPopupDialog = this.roomUserInfoDrawerPopupDialog;
                    if (userInfoDrawerPopupDialog != null && userInfoDrawerPopupDialog.isShow()) {
                        this.roomUserInfoDrawerPopupDialog.changeCaptainStatus();
                    }
                    UserDrawerPopupDialog userDrawerPopupDialog = this.rightRoomUserDialog;
                    if (userDrawerPopupDialog != null && userDrawerPopupDialog.isShow()) {
                        this.rightRoomUserDialog.delayDismiss(500L);
                    }
                    UserDrawerPopupDialog userDrawerPopupDialog2 = this.rightHallUserDialog;
                    if (userDrawerPopupDialog2 == null || !userDrawerPopupDialog2.isShow()) {
                        return;
                    }
                    this.rightHallUserDialog.delayDismiss(500L);
                    return;
                }
                if ("roomMessage".equals(asString)) {
                    this.mMessages.add((PlayLineSocketModel) this.gson.fromJson(str, PlayLineSocketModel.class));
                    this.mMessageAdapter.notifyDataSetChanged();
                    if (this.isRoomOrWorld == 0) {
                        this.mRecyclerMessage.smoothScrollToPosition(this.mMessages.size() - 1);
                        return;
                    }
                    return;
                }
                if ("worldMessage".equals(asString)) {
                    this.mWorldMessage.add((PlayLineSocketModel) this.gson.fromJson(str, PlayLineSocketModel.class));
                    this.mWorldMessageAdapter.notifyDataSetChanged();
                    if (this.isRoomOrWorld == 1) {
                        this.mRecyclerMessage.smoothScrollToPosition(this.mWorldMessage.size() - 1);
                        return;
                    }
                    return;
                }
                if ("enterRoomApply".equals(asString)) {
                    String languageByKey = LanguageUtil.getLanguageByKey(this, "applyEnter");
                    if (asJsonObject.get("userName") != null) {
                        str2 = asJsonObject.get("userName").getAsString();
                    }
                    DialogUtil.showConfirmDialog(this, null, str2 + " " + languageByKey, null, "agreeButton", new View.OnClickListener() { // from class: com.hismith.hismithapp.LineRoomActivity.21
                        @Override // android.view.View.OnClickListener
                        public void onClick(View view) {
                            SocketManager.getInstance().agreeUserRoom(asJsonObject.get("userId") == null ? "" : asJsonObject.get("userId").getAsString());
                            DialogUtil.dismissDialog();
                        }
                    });
                    return;
                }
                if ("roomUser".equals(asString)) {
                    this.mRoomUserList = new ArrayList();
                    this.mRoomUserList.addAll(Arrays.asList(playLineSocketModel.getUser()));
                    UserDrawerPopupDialog userDrawerPopupDialog3 = this.rightRoomUserDialog;
                    if (userDrawerPopupDialog3 != null) {
                        userDrawerPopupDialog3.setUserList(this.mRoomUserList);
                        return;
                    }
                    return;
                }
                if ("hallUser".equals(asString)) {
                    this.mHallUserList = new ArrayList();
                    this.mHallUserList.addAll(Arrays.asList(playLineSocketModel.getUser()));
                    UserDrawerPopupDialog userDrawerPopupDialog4 = this.rightHallUserDialog;
                    if (userDrawerPopupDialog4 != null) {
                        userDrawerPopupDialog4.setUserList(this.mHallUserList);
                        return;
                    }
                    return;
                }
                if ("exitRoom".equals(asString)) {
                    this.appManager.finishActivity(LineRoomActivity.class);
                    finish();
                    return;
                }
                if ("closeRoom".equals(asString)) {
                    ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "roomIsCloseedByHost"));
                    this.appManager.finishActivity(LineRoomActivity.class);
                    finish();
                    return;
                }
                if ("removeRoom".equals(asString)) {
                    ToastUtil.showShortToastCenter(this, LanguageUtil.getLanguageByKey(this, "youOutRoomByHost"));
                    this.appManager.finishActivity(LineRoomActivity.class);
                    finish();
                    return;
                }
                if ("setSpeed".equals(asString)) {
                    closePreviewFragment();
                    if (MyApplication.getApplication().getmUserModel() == null || !this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
                        if (asJsonObject.get("type") != null) {
                            str2 = asJsonObject.get("type").getAsString();
                        }
                        RemotePlayModel remotePlayModel = (RemotePlayModel) this.gson.fromJson(str, RemotePlayModel.class);
                        if (remotePlayModel.getLines() != null && remotePlayModel.getLines().isJsonObject()) {
                            JsonObject jsonObject = (JsonObject) remotePlayModel.getLines();
                            HashMap hashMap = new HashMap();
                            for (String str3 : jsonObject.keySet()) {
                                JsonObject asJsonObject2 = jsonObject.getAsJsonObject(str3);
                                hashMap.put(str3, new Point(asJsonObject2.get("x").getAsInt(), asJsonObject2.get("y").getAsInt()));
                                LogUtil.e("WebSocket line key:" + str3 + ", value:" + asJsonObject2);
                            }
                            if (ExifInterface.GPS_MEASUREMENT_INTERRUPTED.equals(str2)) {
                                this.mPlayViewVibrate.setPointArr(hashMap);
                            } else {
                                this.mPlayView.setPointArr(hashMap);
                            }
                        }
                        if (ExifInterface.GPS_MEASUREMENT_INTERRUPTED.equals(str2)) {
                            int speed = remotePlayModel.getSpeed();
                            int i = speed > 100 ? 100 : speed < 0 ? 0 : speed;
                            this.lastVibrate = i;
                            int i2 = this.mSafeVibrateMax;
                            if (i > i2) {
                                i = i2;
                            }
                            BleManager.getInstance(this).writeVibrate(i);
                            this.mProgressVibrate.setSecondaryProgress(this.lastVibrate);
                            return;
                        }
                        int speed2 = remotePlayModel.getSpeed();
                        int i3 = speed2 > 100 ? 100 : speed2 < 0 ? 0 : speed2;
                        this.lastSpeed = i3;
                        int i4 = this.mSafeMax;
                        if (i3 > i4) {
                            i3 = i4;
                        }
                        LogUtil.e("SocketManager 接收速度执行 mSafeMax：" + this.mSafeMax + ", speed:" + i3);
                        BleManager.getInstance(this).writeSpeed(i3);
                        this.mProgressSpeed.setSecondaryProgress(this.lastSpeed);
                        return;
                    }
                    return;
                }
                if (!"roomSetMode".equals(asString)) {
                    if ("roomStopMode".equals(asString)) {
                        PreviewFragment2 previewFragment2 = this.mPreviewFragment;
                        if (previewFragment2 != null) {
                            previewFragment2.setPlay(false);
                            return;
                        }
                        return;
                    }
                    if ("setStrokeType".equals(asString)) {
                        if (playLineSocketModel.getStrokeType() == 1) {
                            this.runMode = false;
                            BleManager.getInstance(this).writeRunManual();
                            return;
                        } else {
                            this.runMode = true;
                            BleManager.getInstance(this).writeRunAuto();
                            return;
                        }
                    }
                    if ("setStrokeNow".equals(asString)) {
                        this.recePosition = playLineSocketModel.getStrokeNow();
                        int startPosition = BleManager.getInstance(this).getmConnBleDevice().getStartPosition();
                        int stopPosition = BleManager.getInstance(this).getmConnBleDevice().getStopPosition() - startPosition;
                        BleManager.getInstance(this).writePosition(this.recePosition);
                        this.progress_position.setSecondaryProgress(startPosition + (this.recePosition / stopPosition));
                        return;
                    }
                    return;
                }
                int asInt = asJsonObject.get("modeId") == null ? 0 : asJsonObject.get("modeId").getAsInt();
                if (this.mModeingId == asInt && this.isModeing && (MyApplication.getApplication().getmUserModel() == null || this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + ""))) {
                    return;
                }
                this.isModeing = true;
                this.mPlayView.setTouch(false);
                this.mPlayViewVibrate.setTouch(false);
                if (asJsonObject.get("modeData") != null) {
                    asJsonObject.get("modeData").getAsString();
                }
                if (asJsonObject.get("modePoints") != null) {
                    str2 = asJsonObject.get("modePoints").getAsString();
                }
                float asFloat = (asJsonObject.get("playTime") == null ? 0.0f : asJsonObject.get("playTime").getAsFloat()) * 1000.0f;
                if (this.mModeingId != asInt) {
                    this.mModeingId = asInt;
                    initPreviewFragment(asInt, str2, (int) asFloat, 0);
                    return;
                }
                PreviewFragment2 previewFragment22 = this.mPreviewFragment;
                if (previewFragment22 == null) {
                    initPreviewFragment(asInt, str2, (int) asFloat, 0);
                } else {
                    previewFragment22.setmPlayTime((int) asFloat);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-LineRoomActivity, reason: not valid java name */
    /* synthetic */ void m562x21c470aa(Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获得权限");
            LineAudioManager.getInstance().initRecord();
            openMic();
        } else if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选”不再询问“，在这里应该弹出对话框向用户解释为何需要该权限");
        } else {
            ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "microPermission"));
            LogUtil.e("权限申请失败，用户勾选了“不再询问”，在这里应该引导用户去设置页面打开权限");
        }
    }

    @Override // androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, android.app.Activity
    protected void onActivityResult(int i, int i2, Intent intent) {
        super.onActivityResult(i, i2, intent);
    }

    @Override // com.hismith.hismithapp.manager.AudioSocketManager.OnAudioSocketNotifyListener
    public void onAudioClosed(int i, String str) {
        socketAudioFailure();
    }

    @Override // com.hismith.hismithapp.manager.AudioSocketManager.OnAudioSocketNotifyListener
    public void onAudioFailure(Throwable th, Response response) {
        socketAudioFailure();
    }

    @Override // com.hismith.hismithapp.manager.AudioSocketManager.OnAudioSocketNotifyListener
    public void onAudioOpen() {
        if (!this.isCloseMic) {
            LineAudioManager.getInstance().startAudio();
        }
        this.mAudioReconnCount = 0;
        if (this.mHandler.hasMessages(50)) {
            this.mHandler.removeMessages(50);
        }
        this.mHandler.sendEmptyMessage(49);
    }

    @Override // com.hismith.hismithapp.manager.LineAudioManager.OnAudioSpeedChangeListener
    public void onAudioSpeedChange(double d, int i) {
    }

    @Override // com.hismith.hismithapp.receiver.NetBroadcastReceiver.NetChangeListener
    public void onChangeListener(int i) {
        if (i == 2) {
            LogUtil.e("SocketManager 网络状态：WIFI");
            int i2 = this.mLastNetStatus;
            if (i2 != -1 && i2 != 2) {
                SocketManager.getInstance().connSocket(2);
                if (!TextUtils.isEmpty(this.isRoomVoice) && "1".equals(this.isRoomVoice)) {
                    AudioSocketManager.getInstance().connSocket(3, this.isCloseSpeaker);
                }
            }
            this.mLastNetStatus = 2;
            return;
        }
        if (i != 1) {
            if (i == 0) {
                LogUtil.e("SocketManager 网络状态：没网络");
                this.mLastNetStatus = 0;
                return;
            }
            return;
        }
        LogUtil.e("SocketManager 网络状态：移动网络");
        int i3 = this.mLastNetStatus;
        if (i3 != -1 && i3 != 1) {
            SocketManager.getInstance().connSocket(2);
            if (!TextUtils.isEmpty(this.isRoomVoice) && "1".equals(this.isRoomVoice)) {
                AudioSocketManager.getInstance().connSocket(3, this.isCloseSpeaker);
            }
        }
        this.mLastNetStatus = 1;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        int id = view.getId();
        if (id == R.id.view_network_error) {
            LogUtil.e("重新连接 mDisconnType:" + this.mDisconnType);
            if ((this.mDisconnType & 1) == 1) {
                LogUtil.e("重新连接 main:" + (1 & this.mDisconnType));
                this.mReconnCount = 0;
                SocketManager.getInstance().connSocket(2);
            }
            if ((this.mDisconnType & 2) == 2) {
                LogUtil.e("重新连接 audio:" + (2 & this.mDisconnType));
                this.mAudioReconnCount = 0;
                if (!TextUtils.isEmpty(this.isRoomVoice) && "1".equals(this.isRoomVoice)) {
                    AudioSocketManager.getInstance().connSocket(3, this.isCloseSpeaker);
                }
            }
            runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.LineRoomActivity.24
                @Override // java.lang.Runnable
                public void run() {
                    if (LineRoomActivity.this.mTxtNetworkError != null) {
                        LineRoomActivity.this.mViewNetworkError.setVisibility(8);
                        LineRoomActivity lineRoomActivity = LineRoomActivity.this;
                        LanguageUtil.setText(lineRoomActivity, lineRoomActivity.mTxtNetworkError, "notNetwork");
                    }
                }
            });
            return;
        }
        if (id == R.id.line_room_img_owner_head) {
            try {
                IntentUtil.startUserInfoActivity(this, Integer.valueOf(SocketManager.getInstance().getRoomInfoModel().getUserId()).intValue());
                return;
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }
        if (id == R.id.line_room_img_sound) {
            if (this.isCloseSpeaker) {
                LogUtil.e("打开喇叭");
                this.isCloseSpeaker = false;
                this.mImgSound.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_line_sound));
                if (TextUtils.isEmpty(this.isRoomVoice) || !"1".equals(this.isRoomVoice)) {
                    return;
                }
                AudioSocketManager.getInstance().openSound();
                return;
            }
            LogUtil.e("关闭喇叭");
            this.isCloseSpeaker = true;
            this.mImgSound.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_line_sound_close));
            if (TextUtils.isEmpty(this.isRoomVoice) || !"1".equals(this.isRoomVoice)) {
                return;
            }
            AudioSocketManager.getInstance().closeSound();
            return;
        }
        if (id == R.id.line_room_img_voice) {
            if (this.isCloseMic) {
                if (!checkRecordPermission()) {
                    initRecordPermission();
                    return;
                } else {
                    LogUtil.e("已有权限");
                    openMic();
                    return;
                }
            }
            LogUtil.e("关闭麦克风");
            this.isCloseMic = true;
            this.mImgVoice.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_line_voice_close));
            LineAudioManager.getInstance().stopAudio();
            if (TextUtils.isEmpty(this.isRoomVoice) || !"1".equals(this.isRoomVoice)) {
                return;
            }
            AudioSocketManager.getInstance().closeVoice();
            return;
        }
        if (id == R.id.line_room_img_chat) {
            this.isShowMessage = !this.isShowMessage;
            initChatLayout();
            return;
        }
        if (id == R.id.line_room_img_exit) {
            showExitDialog();
            return;
        }
        if (id == R.id.line_room_layout_user_count) {
            initRightRoomUserDialog();
            return;
        }
        if (id == R.id.line_room_img_user_invite) {
            initRightHallUserDialog();
            return;
        }
        if (id == R.id.line_room_img_mode) {
            initModeDialog();
            return;
        }
        if (id == R.id.line_room_img_start) {
            rocketAnimation();
        } else if (id == R.id.img_run_mode) {
            if (this.runMode) {
                BleManager.getInstance(this).writeRunManual();
            } else {
                BleManager.getInstance(this).writeRunAuto();
            }
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onClosed(int i, String str) {
        socketFailure();
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onCodeNull() {
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.LineRoomActivity.2
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                if (LineRoomActivity.this.rightRoomUserDialog != null && LineRoomActivity.this.rightRoomUserDialog.isShow()) {
                    LineRoomActivity.this.rightRoomUserDialog.delayDismiss(500L);
                    return;
                }
                if (LineRoomActivity.this.rightHallUserDialog != null && LineRoomActivity.this.rightHallUserDialog.isShow()) {
                    LineRoomActivity.this.rightHallUserDialog.delayDismiss(500L);
                } else if (LineRoomActivity.this.mModePagerDialog == null || !LineRoomActivity.this.mModePagerDialog.isShow()) {
                    LineRoomActivity.this.showExitDialog();
                } else {
                    LineRoomActivity.this.mModePagerDialog.delayDismiss(500L);
                }
            }
        });
        getWindow().addFlags(128);
        setContentView(R.layout.activity_line_room);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        this.mHandler.removeCallbacksAndMessages(null);
        SocketManager.getInstance().setOnSocketNotifyListener(null);
        AudioSocketManager.getInstance().setOnAudioSocketNotifyListener(null);
        LineAudioManager.getInstance().stopAudio();
        LineAudioManager.getInstance().stopAudioTrack();
        LineAudioManager.getInstance().release();
        SocketManager.getInstance().exitRoom();
        AudioSocketManager.getInstance().clearHandler();
        AudioSocketManager.getInstance().close();
        AudioSocketManager.getInstance().closeSocket();
        BleManager.getInstance(this).setOnConnectStateListener(null);
        BleManager.getInstance(this).writeShutdown();
        BleManager.getInstance(this).setWriteSpeed(true);
        AudioManager audioManager = this.mAudioManager;
        if (audioManager != null) {
            audioManager.abandonAudioFocus(null);
        }
        destroyFloatWindow();
        unregisterReceiver();
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onFailure(Throwable th, Response response) {
        socketFailure();
    }

    @Override // androidx.appcompat.app.AppCompatActivity, android.app.Activity, android.view.KeyEvent.Callback
    public boolean onKeyDown(int i, KeyEvent keyEvent) {
        if (i == 24) {
            this.mAudioManager.adjustStreamVolume(3, 1, 5);
            return true;
        }
        if (i != 25) {
            return super.onKeyDown(i, keyEvent);
        }
        this.mAudioManager.adjustStreamVolume(3, -1, 5);
        return true;
    }

    @Override // com.hismith.hismithapp.view.PlayView_Four.OnLineChangeListener
    public void onLineChange(int i, JsonObject jsonObject) {
        if (i > 100) {
            i = 100;
        } else if (i < 0) {
            i = 0;
        }
        int i2 = this.mSafeMax;
        if (i > i2) {
            i = i2;
        }
        if (i != this.mProgressSpeed.getProgress() || i == 0) {
            this.lastSpeed = i;
            if (BleManager.getInstance(this).getmConnBleDevice() == null || BleManager.getInstance(this).getmConnBleDevice().getRunMode() != 1) {
                BleManager.getInstance(this).writeSpeed(i);
            } else {
                BleManager.getInstance(this).writeRunAuto();
                this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.LineRoomActivity.38
                    @Override // java.lang.Runnable
                    public void run() {
                        BleManager.getInstance(LineRoomActivity.this).writeSpeed(LineRoomActivity.this.lastSpeed);
                    }
                }, 25L);
            }
            SocketManager.getInstance().setSpeed(i, jsonObject);
        }
        SoftKeyboardUtil.hideSoftKeyboard(this, this.mPlayView);
        if (this.mEditMessage.hasFocus()) {
            this.mEditMessage.clearFocus();
        }
    }

    @Override // com.hismith.hismithapp.http.SocketManager.OnSocketNotifyListener
    public void onOpen() {
        this.mReconnCount = 0;
        if (this.mHandler.hasMessages(50)) {
            this.mHandler.removeMessages(50);
        }
        this.mHandler.sendEmptyMessage(49);
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onPause() {
        super.onPause();
        LogUtil.e("onPause");
        try {
            if (this.mEditMessage.hasFocus()) {
                this.mEditMessage.clearFocus();
            }
            if (this.appManager.getActivity(LineRoomActivity.class) != null) {
                showFloatWindow();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onResume() {
        super.onResume();
        destroyFloatWindow();
    }

    @Override // androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onStart() {
        super.onStart();
        initBleDataUI();
    }

    @Override // androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onStop() {
        super.onStop();
        LogUtil.e("onStop");
    }

    /* JADX WARN: Code restructure failed: missing block: B:11:0x0013, code lost:
    
        if (r0 != 3) goto L43;
     */
    @Override // android.view.View.OnTouchListener
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public boolean onTouch(View view, MotionEvent motionEvent) {
        if (!this.isSafeTouch) {
            return false;
        }
        int action = motionEvent.getAction();
        if (action != 0) {
            if (action != 1) {
                if (action == 2) {
                    int top = (int) (view.getTop() + (motionEvent.getY() - this.downY));
                    if (top < 0) {
                        top = 0;
                    } else if (top > this.mProgressSpeed.getHeight()) {
                        top = this.mProgressSpeed.getHeight();
                    }
                    int height = view.getHeight() + top;
                    LogUtil.e("onTouch  t:" + top + ", b:" + height + ", offset:" + (height - top));
                    RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) this.mLayoutMaxSpeed.getLayoutParams();
                    layoutParams.setMargins(0, top, 0, 0);
                    this.mLayoutMaxSpeed.setLayoutParams(layoutParams);
                    int height2 = 100 - (((int) (top * 100.0d)) / this.mProgressSpeed.getHeight());
                    this.mTxtMaxSpeed.setText(height2 + "");
                    this.mSafeMax = height2;
                    SharedPreferencesUtil.put(this, Constans.SAFE_FUN1, Integer.valueOf(height2));
                    PreviewFragment2 previewFragment2 = this.mPreviewFragment;
                    if (previewFragment2 != null) {
                        previewFragment2.setmSafe(this.mSafeMax);
                    }
                    LogUtil.e("限制速度  b:" + height2 + ", lastSpeed:" + this.lastSpeed + ", Progress:" + this.mProgressSpeed.getProgress());
                    if (height2 < 0 || height2 >= this.lastSpeed) {
                        int i = this.lastSpeed;
                        if (height2 >= i && i != this.mProgressSpeed.getProgress() && !this.isModeing) {
                            BleManager.getInstance(this).writeSpeed(this.lastSpeed);
                        }
                    } else if (height2 != this.mProgressSpeed.getProgress()) {
                        if (MyApplication.getApplication().getmUserModel() == null || !this.captainId.equals(MyApplication.getApplication().getmUserModel().getUserId() + "")) {
                            BleManager.getInstance(this).writeSpeed(height2);
                        } else {
                            this.mPlayView.onPointArrChange(height2, null);
                        }
                    }
                }
            }
            LogUtil.e("onTouch 松开Y:" + motionEvent.getY());
        } else {
            this.downY = motionEvent.getY();
            LogUtil.e("onTouch 按下Y:" + motionEvent.getY());
            LogUtil.e("onTouch top:" + view.getTop());
        }
        return true;
    }
}
