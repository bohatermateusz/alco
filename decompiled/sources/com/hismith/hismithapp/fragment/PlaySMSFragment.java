package com.hismith.hismithapp.fragment;

import android.animation.Animator;
import android.animation.AnimatorSet;
import android.animation.ObjectAnimator;
import android.animation.ValueAnimator;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.view.animation.LinearInterpolator;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import androidx.core.content.ContextCompat;
import androidx.media3.extractor.text.ttml.TtmlNode;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.ControlAcceptActivity2;
import com.hismith.hismithapp.ControlSponsorActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.SocketIplayModeAdapter;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.http.SocketManager;
import com.hismith.hismithapp.manager.RecorderVolumeManager;
import com.hismith.hismithapp.model.SocketProductModel;
import com.hismith.hismithapp.model.SocketSpeedModel;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CustomSeekbar;
import com.hismith.hismithapp.view.PlayView_Four;
import com.tbruyelle.rxpermissions2.Permission;
import com.tbruyelle.rxpermissions2.RxPermissions;
import io.reactivex.disposables.Disposable;
import io.reactivex.functions.Consumer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySMSFragment extends SocketPlayFragment implements View.OnClickListener {
    private static final long DOUBLE_CLICK_DELAY = 500;
    private BaseActivity baseActivity;
    private float endDownX;
    private ImageView img_close_left;
    private ImageView img_close_right;
    private ImageView img_run_mode;
    private ImageView iplay_img_close_mode;
    private ImageView iplay_img_close_vibration;
    private ImageView iplay_img_mic;
    private ImageView iplay_img_mode;
    private ImageView iplay_img_position;
    private RelativeLayout iplay_layout;
    private FrameLayout iplay_layout_arrow_down;
    private FrameLayout iplay_layout_arrow_up;
    private FrameLayout iplay_layout_mic_btn;
    private FrameLayout iplay_layout_mode;
    private FrameLayout iplay_layout_mode_btn;
    private FrameLayout iplay_layout_touch_position;
    private FrameLayout iplay_layout_vibrate_arrow_down;
    private FrameLayout iplay_layout_vibrate_arrow_up;
    private FrameLayout iplay_layout_vibration;
    private PlayView_Four iplay_playView;
    private RecyclerView iplay_recycler_mode;
    private RecyclerView iplay_recycler_vibration;
    private FrameLayout layout_position_max;
    private FrameLayout layout_position_min;
    private SocketIplayModeAdapter mAdapter;
    private AnimatorSet mAnimatorSet;
    private Gson mGson;
    private GridLayoutManager mLayoutManager;
    private List<String> mModes;
    private int mPosition;
    private SocketProductModel mProductModel;
    private ProgressBar progress_position;
    private ProgressBar progress_speed;
    private RecorderVolumeManager recorderVolumeManager;
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
    private View view_touch;
    private int mMode = -1;
    private int mSpeed = 0;
    private boolean runMode = false;
    private boolean isMicMode = false;
    private int leftAction = 1;
    private int rightAction = 1;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.1
        /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            int intValue = ((Integer) message.obj).intValue();
            int i = message.what;
            if (i != 1) {
                if (i == 3) {
                    PlaySMSFragment.this.refreshSpeedUI(intValue);
                } else if (i != 4) {
                    switch (i) {
                        case 7:
                            PlaySMSFragment.this.refreshStartBitUI(intValue);
                            break;
                        case 8:
                            PlaySMSFragment.this.refreshEndBitUI(intValue);
                            break;
                        case 9:
                            PlaySMSFragment.this.refreshSmoothUI(intValue);
                            break;
                        case 10:
                            PlaySMSFragment.this.refreshPositionUI(intValue);
                            break;
                    }
                } else {
                    PlaySMSFragment.this.refreshSpeedModeUI(intValue);
                }
            } else if (intValue == 0) {
                PlaySMSFragment.this.runMode = true;
                PlaySMSFragment.this.img_run_mode.setImageResource(R.mipmap.ic_auto_switch_on);
                PlaySMSFragment.this.iplay_layout_touch_position.setVisibility(8);
                PlaySMSFragment.this.iplay_playView.setTouch(true);
            } else if (intValue == 1) {
                PlaySMSFragment.this.runMode = false;
                PlaySMSFragment.this.img_run_mode.setImageResource(R.mipmap.ic_auto_switch_off);
                PlaySMSFragment.this.iplay_layout_touch_position.setVisibility(0);
                PlaySMSFragment.this.iplay_playView.setTouch(false);
            }
            return false;
        }
    });
    private long lastLeftClickTime = 0;
    private long lastRightClickTime = 0;
    private View.OnTouchListener leftOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.7
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMSFragment.this.leftAction = 0;
                PlaySMSFragment.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySMSFragment.this.lastLeftClickTime;
                PlaySMSFragment playSMSFragment = PlaySMSFragment.this;
                if (j < 500) {
                    boolean z = playSMSFragment.baseActivity instanceof ControlSponsorActivity2;
                    PlaySMSFragment playSMSFragment2 = PlaySMSFragment.this;
                    if (z) {
                        ((ControlSponsorActivity2) playSMSFragment2.baseActivity).exit();
                    } else if (playSMSFragment2.baseActivity instanceof ControlAcceptActivity2) {
                        ((ControlAcceptActivity2) PlaySMSFragment.this.baseActivity).exit();
                    }
                    return true;
                }
                playSMSFragment.lastLeftClickTime = currentTimeMillis;
                PlaySMSFragment.this.leftAction = 1;
            }
            return true;
        }
    };
    private View.OnTouchListener rightOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.8
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                PlaySMSFragment.this.rightAction = 0;
                PlaySMSFragment.this.exitIPlay();
            } else if (action == 1) {
                long currentTimeMillis = System.currentTimeMillis();
                long j = currentTimeMillis - PlaySMSFragment.this.lastRightClickTime;
                PlaySMSFragment playSMSFragment = PlaySMSFragment.this;
                if (j < 500) {
                    boolean z = playSMSFragment.baseActivity instanceof ControlSponsorActivity2;
                    PlaySMSFragment playSMSFragment2 = PlaySMSFragment.this;
                    if (z) {
                        ((ControlSponsorActivity2) playSMSFragment2.baseActivity).exit();
                    } else if (playSMSFragment2.baseActivity instanceof ControlAcceptActivity2) {
                        ((ControlAcceptActivity2) PlaySMSFragment.this.baseActivity).exit();
                    }
                    return true;
                }
                playSMSFragment.lastRightClickTime = currentTimeMillis;
                PlaySMSFragment.this.rightAction = 1;
            }
            return true;
        }
    };
    private int mPositionEnd = 100;
    private View.OnTouchListener endOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.9
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                PlaySMSFragment.this.endDownX = motionEvent.getX();
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - PlaySMSFragment.this.endDownX;
            int right = PlaySMSFragment.this.progress_position.getRight() + (PlaySMSFragment.this.layout_position_max.getWidth() / 2);
            LogUtil.e("getRight:" + PlaySMSFragment.this.layout_position_max.getRight());
            int right2 = (int) (right - (PlaySMSFragment.this.layout_position_max.getRight() + x));
            LogUtil.e("onTouch moveX:" + x + ", right():" + right2);
            if (right2 < 0) {
                right2 = 0;
            } else if (right2 > PlaySMSFragment.this.progress_position.getWidth()) {
                right2 = PlaySMSFragment.this.progress_position.getWidth();
            }
            int width = (int) (100.0d - ((right2 * 100.0d) / PlaySMSFragment.this.progress_position.getWidth()));
            if (width <= PlaySMSFragment.this.mPositionStart + 10) {
                width = PlaySMSFragment.this.mPositionStart + 10;
                right2 = (int) (PlaySMSFragment.this.progress_position.getWidth() - ((PlaySMSFragment.this.progress_position.getWidth() / 100.0d) * width));
            }
            if (width > 100) {
                width = 100;
            } else if (width < 0) {
                width = 0;
            }
            RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) PlaySMSFragment.this.layout_position_max.getLayoutParams();
            layoutParams.setMargins(0, 0, right2, 0);
            PlaySMSFragment.this.layout_position_max.setLayoutParams(layoutParams);
            PlaySMSFragment.this.mPositionEnd = width;
            LogUtil.e("结束位置：" + PlaySMSFragment.this.mPositionEnd + ", right:" + right2);
            PlaySMSFragment.this.txt_position_max.setText(width + "");
            SocketManager.getInstance().setStroke(PlaySMSFragment.this.mPositionStart, PlaySMSFragment.this.mPositionEnd);
            PlaySMSFragment.this.getBinderProduct().setStrokeMin(PlaySMSFragment.this.mPositionStart);
            PlaySMSFragment.this.getBinderProduct().setStrokeMax(PlaySMSFragment.this.mPositionEnd);
            PlaySMSFragment playSMSFragment = PlaySMSFragment.this;
            playSMSFragment.refreshEndBitUI(playSMSFragment.mPositionEnd);
            return true;
        }
    };
    private int mPositionStart = 0;
    private View.OnTouchListener startOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.10
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action == 0) {
                view.bringToFront();
                PlaySMSFragment.this.startDownX = motionEvent.getX();
                LogUtil.e("onTouch 按下X:" + motionEvent.getX());
                LogUtil.e("onTouch left:" + view.getLeft());
                return true;
            }
            if (action != 2) {
                return true;
            }
            float x = motionEvent.getX() - PlaySMSFragment.this.startDownX;
            int width = PlaySMSFragment.this.progress_position.getWidth();
            int left = (int) (PlaySMSFragment.this.layout_position_min.getLeft() + x);
            LogUtil.e("onTouch moveX:" + x + ", left():" + left);
            if (left < 0) {
                left = 0;
            } else if (left > width) {
                left = width;
            }
            double d = width;
            int i = (int) ((left * 100.0d) / d);
            if (i >= PlaySMSFragment.this.mPositionEnd - 10) {
                i = PlaySMSFragment.this.mPositionEnd - 10;
                left = (int) ((d / 100.0d) * i);
            }
            if (i > 100) {
                i = 100;
            } else if (i < 0) {
                i = 0;
            }
            RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) PlaySMSFragment.this.layout_position_min.getLayoutParams();
            layoutParams.setMargins(left, 0, 0, 0);
            PlaySMSFragment.this.layout_position_min.setLayoutParams(layoutParams);
            PlaySMSFragment.this.mPositionStart = i;
            LogUtil.e("开始位置：" + PlaySMSFragment.this.mPositionStart + ", left:" + left);
            PlaySMSFragment.this.txt_position_min.setText(i + "");
            SocketManager.getInstance().setStroke(PlaySMSFragment.this.mPositionStart, PlaySMSFragment.this.mPositionEnd);
            PlaySMSFragment.this.getBinderProduct().setStrokeMin(PlaySMSFragment.this.mPositionStart);
            PlaySMSFragment.this.getBinderProduct().setStrokeMax(PlaySMSFragment.this.mPositionEnd);
            PlaySMSFragment playSMSFragment = PlaySMSFragment.this;
            playSMSFragment.refreshStartBitUI(playSMSFragment.mPositionStart);
            return true;
        }
    };
    private View.OnTouchListener positionOnTouch = new View.OnTouchListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.11
        @Override // android.view.View.OnTouchListener
        public boolean onTouch(View view, MotionEvent motionEvent) {
            int action = motionEvent.getAction();
            if (action != 0 && action != 2) {
                return true;
            }
            PlaySMSFragment playSMSFragment = PlaySMSFragment.this;
            int changeUIPosition = playSMSFragment.changeUIPosition(playSMSFragment.iplay_layout_touch_position, PlaySMSFragment.this.iplay_img_position, motionEvent.getX(), motionEvent.getY());
            if (changeUIPosition > 100) {
                changeUIPosition = 100;
            } else if (changeUIPosition < 0) {
                changeUIPosition = 0;
            }
            SocketManager.getInstance().setStrokeNow(changeUIPosition);
            return true;
        }
    };
    private boolean inAnim = false;
    private int isShowMode = 0;
    private int animationDuration = 300;

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
        return 100 - ((int) (((height * 1.0f) / (this.iplay_layout_touch_position.getHeight() - view2.getHeight())) * 100.0f));
    }

    private void closeSocket() {
        if (SocketManager.getInstance().getWebSocket() != null) {
            LogUtil.e("WebSocket 发送：关闭");
            SocketManager.getInstance().close();
            SocketManager.getInstance().closeSocket();
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void exitIPlay() {
        if (this.leftAction == 0 && this.rightAction == 0) {
            BaseActivity baseActivity = this.baseActivity;
            if (baseActivity instanceof ControlSponsorActivity2) {
                ((ControlSponsorActivity2) baseActivity).exit();
            } else if (baseActivity instanceof ControlAcceptActivity2) {
                ((ControlAcceptActivity2) baseActivity).exit();
            }
        }
        if (this.leftAction == 1) {
            LogUtil.e("onTouch 左抖动");
            shakeAnimation(this.img_close_left);
            ToastUtil.showShortToastCenter(this.baseActivity, LanguageUtil.getLanguageByKey(this.baseActivity, "exitDoubleGameNote"));
        }
        if (this.rightAction == 1) {
            LogUtil.e("onTouch 右抖动");
            shakeAnimation(this.img_close_right);
            ToastUtil.showShortToastCenter(this.baseActivity, LanguageUtil.getLanguageByKey(this.baseActivity, "exitDoubleGameNote"));
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public SocketProductModel getBinderProduct() {
        BaseActivity baseActivity = this.baseActivity;
        return baseActivity instanceof ControlSponsorActivity2 ? ((ControlSponsorActivity2) baseActivity).getBinder().getSocketProductModel() : baseActivity instanceof ControlAcceptActivity2 ? ((ControlAcceptActivity2) baseActivity).getBinder().getSocketProductModel() : new SocketProductModel();
    }

    private void init() {
        this.mGson = new Gson();
        Bundle arguments = getArguments();
        if (arguments != null) {
            this.mProductModel = (SocketProductModel) arguments.getSerializable("ProductModel");
        }
        this.recorderVolumeManager = new RecorderVolumeManager();
        GridLayoutManager gridLayoutManager = new GridLayoutManager(this.baseActivity, 2);
        this.mLayoutManager = gridLayoutManager;
        this.iplay_recycler_mode.setLayoutManager(gridLayoutManager);
        this.mModes = new ArrayList();
        SocketIplayModeAdapter socketIplayModeAdapter = new SocketIplayModeAdapter(this.baseActivity, this.mModes);
        this.mAdapter = socketIplayModeAdapter;
        socketIplayModeAdapter.setOnItemClickListener(new SocketIplayModeAdapter.OnItemClickListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.3
            @Override // com.hismith.hismithapp.adapter.SocketIplayModeAdapter.OnItemClickListener
            public void onItemClick(int i) {
                PlaySMSFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode_select);
                int i2 = i + 1;
                SocketManager.getInstance().setMode(i2);
                PlaySMSFragment.this.getBinderProduct().setMode(i2);
            }
        });
        this.iplay_recycler_mode.setAdapter(this.mAdapter);
        if (this.mProductModel != null) {
            initProductData();
            initBleDataUI(this.mProductModel);
        }
    }

    private void initBleDataUI(SocketProductModel socketProductModel) {
        BleManager.getInstance(this.baseActivity).writeShutdown();
        if (socketProductModel != null) {
            Message message = new Message();
            message.what = 1;
            message.obj = Integer.valueOf(socketProductModel.getStrokeType());
            this.mHandler.sendMessage(message);
            Message message2 = new Message();
            message2.what = 3;
            message2.obj = Integer.valueOf(socketProductModel.getSpeed());
            this.mHandler.sendMessage(message2);
            Message message3 = new Message();
            message3.what = 4;
            message3.obj = Integer.valueOf(socketProductModel.getMode());
            this.mHandler.sendMessage(message3);
            Message message4 = new Message();
            message4.what = 7;
            message4.obj = Integer.valueOf(socketProductModel.getStrokeMin());
            this.mHandler.sendMessage(message4);
            Message message5 = new Message();
            message5.what = 8;
            message5.obj = Integer.valueOf(socketProductModel.getStrokeMax());
            this.mHandler.sendMessage(message5);
            Message message6 = new Message();
            message6.what = 9;
            message6.obj = Integer.valueOf(socketProductModel.getSmooth());
            this.mHandler.sendMessage(message6);
            Message message7 = new Message();
            message7.what = 10;
            message7.obj = Integer.valueOf(socketProductModel.getStrokeNow());
            this.mHandler.sendMessage(message7);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initModeShow() {
        int i = this.isShowMode;
        if (i != 0) {
            if (i == 1) {
                startAnimation(this.iplay_layout_mode);
            }
        } else {
            FrameLayout frameLayout = this.iplay_layout_mode;
            if (frameLayout == null || frameLayout.getVisibility() != 0) {
                return;
            }
            stopAnimation(this.iplay_layout_mode);
        }
    }

    private void initRecordPermission() {
        if (ContextCompat.checkSelfPermission(this.baseActivity, "android.permission.RECORD_AUDIO") == 0) {
            startMicClick();
            return;
        }
        Disposable subscribe = new RxPermissions(this).requestEachCombined("android.permission.RECORD_AUDIO").subscribe(new Consumer() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment$$ExternalSyntheticLambda0
            @Override // io.reactivex.functions.Consumer
            public final void accept(Object obj) {
                PlaySMSFragment.this.m600xb76ee2c4((Permission) obj);
            }
        });
        if (this.baseActivity.mCompositeDisposable == null || this.baseActivity.mCompositeDisposable.isDisposed()) {
            return;
        }
        this.baseActivity.mCompositeDisposable.add(subscribe);
    }

    private void initView(View view) {
        this.iplay_layout = (RelativeLayout) view.findViewById(R.id.iplay_layout);
        this.img_run_mode = (ImageView) view.findViewById(R.id.img_run_mode);
        this.iplay_layout_mic_btn = (FrameLayout) view.findViewById(R.id.iplay_layout_mic_btn);
        this.iplay_img_mic = (ImageView) view.findViewById(R.id.iplay_img_mic);
        this.iplay_layout_mode_btn = (FrameLayout) view.findViewById(R.id.iplay_layout_mode_btn);
        this.iplay_img_mode = (ImageView) view.findViewById(R.id.iplay_img_mode);
        this.txt_speed_label = (TextView) view.findViewById(R.id.txt_speed_label);
        this.txt_speed = (TextView) view.findViewById(R.id.txt_speed);
        this.progress_speed = (ProgressBar) view.findViewById(R.id.progress_speed);
        this.txt_position_label = (TextView) view.findViewById(R.id.txt_position_label);
        this.txt_position = (TextView) view.findViewById(R.id.txt_position);
        this.progress_position = (ProgressBar) view.findViewById(R.id.progress_position);
        this.layout_position_max = (FrameLayout) view.findViewById(R.id.layout_position_max);
        this.txt_position_max = (TextView) view.findViewById(R.id.txt_position_max);
        this.layout_position_min = (FrameLayout) view.findViewById(R.id.layout_position_min);
        this.txt_position_min = (TextView) view.findViewById(R.id.txt_position_min);
        this.txt_smooth_label = (TextView) view.findViewById(R.id.txt_smooth_label);
        this.txt_smooth = (TextView) view.findViewById(R.id.txt_smooth);
        this.seek_smooth = (CustomSeekbar) view.findViewById(R.id.seek_smooth);
        this.iplay_playView = (PlayView_Four) view.findViewById(R.id.iplay_playView);
        this.iplay_layout_touch_position = (FrameLayout) view.findViewById(R.id.iplay_layout_touch_position);
        this.iplay_img_position = (ImageView) view.findViewById(R.id.iplay_img_position);
        this.iplay_layout_mode = (FrameLayout) view.findViewById(R.id.iplay_layout_mode);
        this.iplay_recycler_mode = (RecyclerView) view.findViewById(R.id.iplay_recycler_mode);
        this.iplay_layout_arrow_up = (FrameLayout) view.findViewById(R.id.iplay_layout_arrow_up);
        this.iplay_layout_arrow_down = (FrameLayout) view.findViewById(R.id.iplay_layout_arrow_down);
        this.iplay_img_close_mode = (ImageView) view.findViewById(R.id.iplay_img_close_mode);
        this.iplay_layout_vibration = (FrameLayout) view.findViewById(R.id.iplay_layout_vibration);
        this.iplay_recycler_vibration = (RecyclerView) view.findViewById(R.id.iplay_recycler_vibration);
        this.iplay_layout_vibrate_arrow_up = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_arrow_up);
        this.iplay_layout_vibrate_arrow_down = (FrameLayout) view.findViewById(R.id.iplay_layout_vibrate_arrow_down);
        this.iplay_img_close_vibration = (ImageView) view.findViewById(R.id.iplay_img_close_vibration);
        this.img_close_left = (ImageView) view.findViewById(R.id.img_close_left);
        this.img_close_right = (ImageView) view.findViewById(R.id.img_close_right);
        this.view_touch = view.findViewById(R.id.view_touch);
        LanguageUtil.setText(this.baseActivity, this.txt_speed_label, "speedTitle");
        this.txt_position_label.setText(LanguageUtil.getLanguageByKey(this.baseActivity, "position") + ":");
        this.txt_smooth_label.setText(LanguageUtil.getLanguageByKey(this.baseActivity, "smoothness") + ":");
        this.iplay_playView.setProgressBar(this.progress_speed);
        this.iplay_playView.setTouch(false);
        this.seek_smooth.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.2
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
                LogUtil.e("onProgressChanged:" + i);
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
                LogUtil.e("onStartTrackingTouch:" + PlaySMSFragment.this.seek_smooth.getProgress());
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
                int i;
                LogUtil.e("onStopTrackingTouch:" + PlaySMSFragment.this.seek_smooth.getProgress());
                int progress = PlaySMSFragment.this.seek_smooth.getProgress();
                int i2 = progress % 100;
                if (i2 < 50) {
                    i = progress - i2;
                    LogUtil.e("newProgress:" + i);
                } else {
                    i = (progress - i2) + 100;
                    LogUtil.e("newProgress:" + i);
                }
                int i3 = i / 100;
                PlaySMSFragment.this.txt_smooth.setText(i3 + "");
                SocketManager.getInstance().setSmooth(i3);
                PlaySMSFragment.this.getBinderProduct().setSmooth(i3);
                ValueAnimator ofInt = ValueAnimator.ofInt(progress, i);
                ofInt.setInterpolator(new LinearInterpolator());
                ofInt.setDuration(300L);
                ofInt.setRepeatCount(0);
                ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.2.1
                    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                    public void onAnimationUpdate(ValueAnimator valueAnimator) {
                        PlaySMSFragment.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
                    }
                });
                ofInt.start();
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshEndBitUI(int i) {
        this.mPositionEnd = i;
        this.txt_position_max.setText(this.mPositionEnd + "");
        double width = this.progress_position.getWidth();
        int i2 = (int) (width - ((width / 100.0d) * i));
        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) this.layout_position_max.getLayoutParams();
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
    public void refreshSmoothUI(int i) {
        this.txt_smooth.setText(i + "");
        ValueAnimator ofInt = ValueAnimator.ofInt(this.seek_smooth.getProgress(), i * 100);
        ofInt.setInterpolator(new LinearInterpolator());
        ofInt.setDuration(300L);
        ofInt.setRepeatCount(0);
        ofInt.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.12
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator) {
                LogUtil.e("refreshSmoothUI:" + ((Integer) valueAnimator.getAnimatedValue()));
                PlaySMSFragment.this.seek_smooth.setProgress(((Integer) valueAnimator.getAnimatedValue()).intValue());
            }
        });
        ofInt.start();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSpeedModeUI(int i) {
        ImageView imageView = this.iplay_img_mode;
        if (i <= 0) {
            imageView.setImageResource(R.mipmap.ic_iplay_mode);
            this.isShowMode = 0;
            if (this.iplay_layout_mode.getVisibility() == 0) {
                stopAnimation(this.iplay_layout_mode);
            }
        } else {
            imageView.setImageResource(R.mipmap.ic_iplay_mode_select);
            this.isShowMode = 1;
            if (this.iplay_layout_mode.getVisibility() != 0) {
                startAnimation(this.iplay_layout_mode);
            }
        }
        SocketIplayModeAdapter socketIplayModeAdapter = this.mAdapter;
        if (socketIplayModeAdapter != null) {
            socketIplayModeAdapter.setClickPosition(i - 1);
            if (i != 0) {
                this.iplay_recycler_mode.smoothScrollToPosition(i);
            }
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshSpeedUI(int i) {
        this.progress_speed.setProgress(i);
        this.txt_speed.setText(i + "");
        refreshSpeedModeUI(0);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshStartBitUI(int i) {
        this.mPositionStart = i;
        this.txt_position_min.setText(this.mPositionStart + "");
        RelativeLayout.LayoutParams layoutParams = (RelativeLayout.LayoutParams) this.layout_position_min.getLayoutParams();
        layoutParams.setMargins((int) ((this.progress_position.getWidth() / 100.0d) * i), 0, 0, 0);
        this.layout_position_min.setLayoutParams(layoutParams);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void scrollModeHideImg() {
        if (this.iplay_recycler_mode.canScrollVertically(1)) {
            LogUtil.e("direction 1: true");
            if (this.iplay_layout_arrow_down.getVisibility() == 4) {
                this.iplay_layout_arrow_down.setVisibility(0);
            }
        } else {
            LogUtil.e("direction 1: false 滑动到底部");
            this.iplay_layout_arrow_down.setVisibility(4);
        }
        if (!this.iplay_recycler_mode.canScrollVertically(-1)) {
            LogUtil.e("direction -1: false 滑动到顶部");
            this.iplay_layout_arrow_up.setVisibility(4);
        } else {
            LogUtil.e("direction -1: true");
            if (this.iplay_layout_arrow_up.getVisibility() == 4) {
                this.iplay_layout_arrow_up.setVisibility(0);
            }
        }
    }

    private void setListener() {
        this.img_run_mode.setOnClickListener(this);
        this.iplay_layout_mic_btn.setOnClickListener(this);
        this.iplay_layout_mode_btn.setOnClickListener(this);
        this.img_close_left.setOnTouchListener(this.leftOnTouch);
        this.img_close_right.setOnTouchListener(this.rightOnTouch);
        this.layout_position_max.setOnTouchListener(this.endOnTouch);
        this.layout_position_min.setOnTouchListener(this.startOnTouch);
        this.iplay_layout_touch_position.setOnTouchListener(this.positionOnTouch);
        this.iplay_playView.setOnLineChangeListener(new PlayView_Four.OnLineChangeListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.4
            @Override // com.hismith.hismithapp.view.PlayView_Four.OnLineChangeListener
            public void onLineChange(int i, JsonObject jsonObject) {
                LogUtil.e("onLineChange speed:" + i);
                if (i < 0) {
                    i = 0;
                } else if (i > 100) {
                    i = 100;
                }
                if (i != PlaySMSFragment.this.progress_speed.getProgress() || i == 0) {
                    PlaySMSFragment.this.txt_speed.setText(i + "");
                    PlaySMSFragment.this.progress_speed.setProgress(i);
                    PlaySMSFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                    if (PlaySMSFragment.this.mMode != 0) {
                        PlaySMSFragment.this.mMode = 0;
                        if (PlaySMSFragment.this.mAdapter != null) {
                            PlaySMSFragment.this.mAdapter.setClickPosition(PlaySMSFragment.this.mMode - 1);
                        }
                    }
                    SocketManager.getInstance().setSpeed(i, jsonObject);
                    PlaySMSFragment.this.getBinderProduct().setSpeed(i);
                    PlaySMSFragment.this.getBinderProduct().setMode(0);
                    PlaySMSFragment.this.mSpeed = i;
                }
            }
        });
        this.recorderVolumeManager.setOnMicStatusListener(new RecorderVolumeManager.OnMicStatusListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.5
            @Override // com.hismith.hismithapp.manager.RecorderVolumeManager.OnMicStatusListener
            public void onMicAmplitude(double d) {
                double d2 = d - 55.0d;
                if (d2 < 0.0d) {
                    d2 = 0.0d;
                }
                int i = (int) (d2 * 2.0d);
                if (i > 100) {
                    i = 100;
                }
                PlaySMSFragment.this.mSpeed = i;
                PlaySMSFragment.this.baseActivity.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.5.1
                    @Override // java.lang.Runnable
                    public void run() {
                        PlaySMSFragment.this.txt_speed.setText(PlaySMSFragment.this.mSpeed + "");
                        PlaySMSFragment.this.progress_speed.setProgress(PlaySMSFragment.this.mSpeed);
                        PlaySMSFragment.this.iplay_img_mode.setImageResource(R.mipmap.ic_iplay_mode);
                        if (PlaySMSFragment.this.mMode != 0) {
                            PlaySMSFragment.this.mMode = 0;
                            if (PlaySMSFragment.this.mAdapter != null) {
                                PlaySMSFragment.this.mAdapter.setClickPosition(PlaySMSFragment.this.mMode - 1);
                            }
                        }
                    }
                });
                SocketManager.getInstance().setSpeed(i);
                PlaySMSFragment.this.getBinderProduct().setSpeed(i);
                PlaySMSFragment.this.getBinderProduct().setMode(0);
            }
        });
        this.iplay_recycler_mode.addOnScrollListener(new RecyclerView.OnScrollListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.6
            @Override // androidx.recyclerview.widget.RecyclerView.OnScrollListener
            public void onScrolled(RecyclerView recyclerView, int i, int i2) {
                super.onScrolled(recyclerView, i, i2);
                PlaySMSFragment.this.scrollModeHideImg();
            }
        });
    }

    private void startAnimation(final FrameLayout frameLayout) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 0.0f, 1.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 0.0f, 1.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.14
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMSFragment.this.inAnim = false;
                if (i != PlaySMSFragment.this.isShowMode) {
                    PlaySMSFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMSFragment.this.inAnim = true;
                frameLayout.setVisibility(0);
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
        this.mAnimatorSet.start();
    }

    private void startMicClick() {
        boolean z = this.isMicMode;
        this.isMicMode = !z;
        ImageView imageView = this.iplay_img_mic;
        if (!z) {
            imageView.setImageResource(R.mipmap.ic_iplay_mic_select);
            this.recorderVolumeManager.startRecording();
            return;
        }
        imageView.setImageResource(R.mipmap.ic_iplay_mic);
        this.recorderVolumeManager.stopRecording();
        SocketManager.getInstance().setSpeed(0);
        getBinderProduct().setSpeed(0);
        getBinderProduct().setMode(0);
        Message message = new Message();
        message.what = 3;
        message.obj = 0;
        this.mHandler.sendMessage(message);
    }

    private void stopAnimation(final FrameLayout frameLayout) {
        if (this.inAnim) {
            return;
        }
        final int i = this.isShowMode;
        ObjectAnimator ofFloat = ObjectAnimator.ofFloat(frameLayout, "scaleX", 1.0f, 0.0f);
        ObjectAnimator ofFloat2 = ObjectAnimator.ofFloat(frameLayout, "scaleY", 1.0f, 0.0f);
        ofFloat.setDuration(this.animationDuration);
        ofFloat2.setDuration(this.animationDuration);
        ofFloat.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.15
            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationCancel(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationEnd(Animator animator) {
                PlaySMSFragment.this.inAnim = false;
                frameLayout.setVisibility(8);
                if (i != PlaySMSFragment.this.isShowMode) {
                    PlaySMSFragment.this.initModeShow();
                }
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationRepeat(Animator animator) {
            }

            @Override // android.animation.Animator.AnimatorListener
            public void onAnimationStart(Animator animator) {
                PlaySMSFragment.this.inAnim = true;
            }
        });
        AnimatorSet animatorSet = new AnimatorSet();
        this.mAnimatorSet = animatorSet;
        animatorSet.playTogether(ofFloat, ofFloat2);
        this.mAnimatorSet.start();
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void initProductData() {
        if (this.mProductModel == null) {
            return;
        }
        PlayView_Four playView_Four = this.iplay_playView;
        if (playView_Four != null) {
            playView_Four.setmScreenSpeed(r0.getSlide());
        }
        this.mModes.clear();
        SocketProductModel.Modes modesUrl = this.mProductModel.getModesUrl();
        FrameLayout frameLayout = this.iplay_layout_mode_btn;
        if (modesUrl != null) {
            frameLayout.setVisibility(0);
            this.mModes.addAll(Arrays.asList(this.mProductModel.getModesUrl().getModes()));
        } else {
            frameLayout.setVisibility(8);
        }
        this.mAdapter.notifyDataSetChanged();
        scrollModeHideImg();
    }

    /* renamed from: lambda$initRecordPermission$0$com-hismith-hismithapp-fragment-PlaySMSFragment, reason: not valid java name */
    /* synthetic */ void m600xb76ee2c4(Permission permission) throws Exception {
        if (permission.granted) {
            LogUtil.e("获取权限成功");
            startMicClick();
        } else if (permission.shouldShowRequestPermissionRationale) {
            LogUtil.e("获取权限失败，但是用户没有勾选“不再询问”");
        } else {
            LogUtil.e("权限申请失败，用户勾选了“不再询问”");
            DialogUtil.showConfirmDialog(this.baseActivity, "prompt", "voicePermissions", null, TtmlNode.TEXT_EMPHASIS_MARK_OPEN, new View.OnClickListener() { // from class: com.hismith.hismithapp.fragment.PlaySMSFragment.13
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    Intent intent = new Intent();
                    intent.setAction("android.settings.APPLICATION_DETAILS_SETTINGS");
                    intent.setData(Uri.parse("package:" + PlaySMSFragment.this.baseActivity.getPackageName()));
                    PlaySMSFragment.this.startActivity(intent);
                    DialogUtil.dismissDialog();
                }
            });
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment, com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        if (view.getId() != R.id.img_run_mode) {
            if (view.getId() != R.id.iplay_layout_mode_btn) {
                if (view.getId() == R.id.iplay_layout_mic_btn) {
                    initRecordPermission();
                    return;
                }
                return;
            } else {
                if (this.inAnim) {
                    return;
                }
                if (this.iplay_layout_mode.getVisibility() == 0) {
                    stopAnimation(this.iplay_layout_mode);
                    return;
                } else {
                    this.isShowMode = 1;
                    startAnimation(this.iplay_layout_mode);
                    return;
                }
            }
        }
        if (this.runMode) {
            Message message = new Message();
            message.what = 1;
            message.obj = 1;
            this.mHandler.sendMessage(message);
            SocketManager.getInstance().setStrokeType(1);
            getBinderProduct().setStrokeType(1);
            return;
        }
        Message message2 = new Message();
        message2.what = 1;
        message2.obj = 0;
        this.mHandler.sendMessage(message2);
        SocketManager.getInstance().setStrokeType(0);
        getBinderProduct().setStrokeType(0);
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_iplay_new, viewGroup, false);
        initView(inflate);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        RecorderVolumeManager recorderVolumeManager = this.recorderVolumeManager;
        if (recorderVolumeManager != null) {
            recorderVolumeManager.cancelRecording();
        }
    }

    @Override // com.hismith.hismithapp.fragment.SocketPlayFragment
    public void onNotify(String str) {
        try {
            SocketSpeedModel socketSpeedModel = (SocketSpeedModel) this.mGson.fromJson(str, SocketSpeedModel.class);
            if (socketSpeedModel != null && "setStrokeNow".equals(socketSpeedModel.getAction()) && socketSpeedModel.getOnlyRead() == 1) {
                refreshPositionUI(socketSpeedModel.getStrokeNow());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void shakeAnimation(View view) {
        view.clearAnimation();
        Animation loadAnimation = AnimationUtils.loadAnimation(this.baseActivity, R.anim.iplay_exit_anim);
        loadAnimation.reset();
        loadAnimation.setFillAfter(true);
        view.startAnimation(loadAnimation);
    }
}
