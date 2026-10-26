package com.hismith.hismithapp;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.SeekBar;
import androidx.activity.OnBackPressedCallback;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.constraintlayout.widget.ConstraintSet;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.OnApplyWindowInsetsListener;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.media3.common.C;
import androidx.media3.common.MediaItem;
import androidx.media3.common.PlaybackException;
import androidx.media3.common.Player;
import androidx.media3.common.VideoSize;
import androidx.media3.exoplayer.ExoPlayer;
import androidx.transition.AutoTransition;
import androidx.transition.TransitionManager;
import com.facebook.appevents.integrity.IntegrityManager;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.bluetooth.BleManager;
import com.hismith.hismithapp.databinding.ActivityPlayVideoBinding;
import com.hismith.hismithapp.dialog.PlayVideoPanelDialog;
import com.hismith.hismithapp.pose.MediaPickerHelper;
import com.hismith.hismithapp.pose.PoseLandmarkerHelper5;
import com.hismith.hismithapp.pose.PoseMoveSpeedTracker;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.ModeTimeUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.FeatureGuideView;
import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/* loaded from: classes3.dex */
public class PlayVideoActivity extends BaseActivity {
    private static final long VIDEO_INTERVAL_MS = 100;
    private static final int WHAT_PLAY_TIME = 417;
    private ActivityPlayVideoBinding binding;
    private PlayVideoPanelDialog mPanelDialog;
    private MediaPickerHelper picker;
    private ExoPlayer player;
    private FeatureGuideView videoPanelTipView;
    private PoseLandmarkerHelper5 videoPoseLandmarkerHelper;
    private boolean videoPanelTipShown = false;
    private final int[][] skeleton = {new int[]{5, 7}, new int[]{7, 9}, new int[]{6, 8}, new int[]{8, 10}, new int[]{11, 13}, new int[]{13, 15}, new int[]{12, 14}, new int[]{14, 16}, new int[]{5, 6}, new int[]{11, 12}, new int[]{5, 11}, new int[]{6, 12}, new int[]{0, 5}, new int[]{0, 6}, new int[]{0, 1}, new int[]{0, 2}, new int[]{1, 3}, new int[]{2, 4}};
    private PoseMoveSpeedTracker moveSpeedTracker = new PoseMoveSpeedTracker();
    private boolean userPaused = true;
    private boolean mIsEnterBluetooth = false;
    private boolean mIsSeeking = false;
    private boolean isCaching = false;
    private int lastMainPersonIndex = -1;
    private float[] lastMainFingerprint = null;
    private int mWidth = 0;
    private int mHeight = 0;
    private long mPlayTime = 0;
    private int safeFuns1 = 40;
    private int safeFuns2 = 40;
    private boolean funs1Control = true;
    private boolean funs2Control = true;
    private int animDuration = 100;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.PlayVideoActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            Bundle data = message.getData();
            if (data != null) {
                data.getString(Constans.BleType);
            }
            int i = message.what;
            if (i == 3) {
                PlayVideoActivity.this.onSpeedUI(((Integer) message.obj).intValue());
                return false;
            }
            if (i != PlayVideoActivity.WHAT_PLAY_TIME || PlayVideoActivity.this.player == null) {
                return false;
            }
            PlayVideoActivity playVideoActivity = PlayVideoActivity.this;
            playVideoActivity.mPlayTime = playVideoActivity.player.getCurrentPosition();
            PlayVideoActivity playVideoActivity2 = PlayVideoActivity.this;
            playVideoActivity2.runVideoResult(playVideoActivity2.mPlayTime);
            PlayVideoActivity.this.mHandler.sendEmptyMessageDelayed(PlayVideoActivity.WHAT_PLAY_TIME, 100L);
            return false;
        }
    });
    int mSpeed = 0;
    int mVibrate = 0;
    private BroadcastReceiver mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.PlayVideoActivity.11
        @Override // android.content.BroadcastReceiver
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (Constans.RECEIVER_BLE_CONN_STATE.equals(action)) {
                final int intExtra = intent.getIntExtra("ble_state", 0);
                PlayVideoActivity.this.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.PlayVideoActivity.11.1
                    @Override // java.lang.Runnable
                    public void run() {
                        int i = intExtra;
                        AnonymousClass11 anonymousClass11 = AnonymousClass11.this;
                        if (i == 2) {
                            PlayVideoActivity.this.binding.imgBluetooth.setImageResource(R.mipmap.ic_play_bluetooth_connect);
                        } else {
                            PlayVideoActivity.this.binding.imgBluetooth.setImageResource(R.mipmap.ic_play_bluetooth_disconnect);
                        }
                    }
                });
                return;
            }
            if (Constans.RECEIVER_PRODUCT_INFO.equals(action)) {
                PlayVideoActivity.this.funs1Control = true;
                PlayVideoActivity.this.funs2Control = true;
                return;
            }
            if (Constans.RECEIVER_BLE_RECE_DATA.equals(action)) {
                String stringExtra = intent.getStringExtra(Constans.BleType);
                String stringExtra2 = intent.getStringExtra(Constans.BleData0);
                String stringExtra3 = intent.getStringExtra(Constans.BleData1);
                String stringExtra4 = intent.getStringExtra(IntegrityManager.INTEGRITY_TYPE_ADDRESS);
                if (TextUtils.isEmpty(stringExtra4) || !stringExtra4.equals(MyApplication.getApplication().getDeviceAddress())) {
                    return;
                }
                if ("AA".equals(stringExtra)) {
                    PlayVideoActivity.this.initBleAAData(stringExtra2, stringExtra3);
                }
                if ("CA".equals(stringExtra) || "CC".equals(stringExtra)) {
                    PlayVideoActivity.this.initBleCCData(stringExtra, stringExtra2, stringExtra3);
                }
            }
        }
    };

    /* renamed from: com.hismith.hismithapp.PlayVideoActivity$3, reason: invalid class name */
    class AnonymousClass3 implements MediaPickerHelper.Callback {
        AnonymousClass3() {
        }

        @Override // com.hismith.hismithapp.pose.MediaPickerHelper.Callback
        public void onError(Exception exc) {
        }

        @Override // com.hismith.hismithapp.pose.MediaPickerHelper.Callback
        public void onImagePicked(Bitmap bitmap, Uri uri) {
        }

        @Override // com.hismith.hismithapp.pose.MediaPickerHelper.Callback
        public void onVideoPicked(File file, Uri uri) {
            PlayVideoActivity.this.userPaused = true;
            PlayVideoActivity.this.maybeShowVideoPanelTip();
            PlayVideoActivity.this.binding.overlay.clear();
            PlayVideoActivity.this.showCachingHint();
            PlayVideoActivity.this.moveSpeedTracker.reset();
            PlayVideoActivity.this.lastMainPersonIndex = -1;
            PlayVideoActivity.this.lastMainFingerprint = null;
            PlayVideoActivity.this.binding.seekVideo.setProgress(0);
            PlayVideoActivity.this.binding.seekVideo.setSecondaryProgress(0);
            PlayVideoActivity.this.binding.txtVideoTimeCurrent.setText(ModeTimeUtil.formatMilliTime(0));
            MediaItem fromUri = MediaItem.fromUri(uri);
            PlayVideoActivity.this.player.setPlayWhenReady(false);
            PlayVideoActivity.this.player.setMediaItem(fromUri);
            PlayVideoActivity.this.player.prepare();
            PlayVideoActivity.this.videoPoseLandmarkerHelper.detectVideoFileAsync(uri, 100L, new PoseLandmarkerHelper5.VideoFrameCallback() { // from class: com.hismith.hismithapp.PlayVideoActivity.3.1
                @Override // com.hismith.hismithapp.pose.PoseLandmarkerHelper5.VideoFrameCallback
                public void onCacheProgress(final int i, final long j) {
                    PlayVideoActivity.this.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.PlayVideoActivity.3.1.2
                        @Override // java.lang.Runnable
                        public void run() {
                            if (PlayVideoActivity.this.binding != null) {
                                LogUtil.e("onCacheProgress cachedFrames" + i + ", cachedUntilMs:" + j);
                                PlayVideoActivity.this.binding.seekVideo.setSecondaryProgress((int) Math.min(j, PlayVideoActivity.this.binding.seekVideo.getMax()));
                            }
                        }
                    });
                }

                @Override // com.hismith.hismithapp.pose.PoseLandmarkerHelper5.VideoFrameCallback
                public void onCanPlayed() {
                    LogUtil.e("onCanPlayed():");
                    PlayVideoActivity.this.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.PlayVideoActivity.3.1.4
                        @Override // java.lang.Runnable
                        public void run() {
                            PlayVideoActivity.this.hideCachingHint();
                            if (PlayVideoActivity.this.player == null || PlayVideoActivity.this.player.isPlaying() || PlayVideoActivity.this.userPaused) {
                                return;
                            }
                            PlayVideoActivity.this.player.play();
                        }
                    });
                }

                @Override // com.hismith.hismithapp.pose.PoseLandmarkerHelper5.VideoFrameCallback
                public void onComplete(final boolean z, int i, long j, final int i2, String str) {
                    LogUtil.e("onComplete():" + z + ", errorCode:" + i2 + ", errorMsg:" + str);
                    PlayVideoActivity.this.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.PlayVideoActivity.3.1.3
                        @Override // java.lang.Runnable
                        public void run() {
                            PlayVideoActivity.this.hideCachingHint();
                            if (z) {
                                return;
                            }
                            int i3 = i2;
                            AnonymousClass1 anonymousClass1 = AnonymousClass1.this;
                            if (i3 == 5) {
                                ToastUtil.showWarnShortToast(PlayVideoActivity.this, "videoFormatNotSupportPose");
                            } else {
                                ToastUtil.showWarnShortToast(PlayVideoActivity.this, "videoCacheFailedRetry");
                            }
                            PlayVideoActivity.this.mSpeed = 0;
                            PlayVideoActivity.this.mVibrate = 0;
                            BleManager.getInstance(PlayVideoActivity.this).writeSpeed(0);
                            if (ProductUtil.isSMVType()) {
                                BleManager.getInstance(PlayVideoActivity.this).writeVibrate(0);
                            }
                            if (PlayVideoActivity.this.player == null || !PlayVideoActivity.this.player.isPlaying()) {
                                return;
                            }
                            PlayVideoActivity.this.player.pause();
                        }
                    });
                }

                @Override // com.hismith.hismithapp.pose.PoseLandmarkerHelper5.VideoFrameCallback
                public void onFrameRatio(final int i, final int i2) {
                    PlayVideoActivity.this.runOnUiThread(new Runnable() { // from class: com.hismith.hismithapp.PlayVideoActivity.3.1.1
                        @Override // java.lang.Runnable
                        public void run() {
                            PlayVideoActivity.this.mWidth = i;
                            PlayVideoActivity.this.mHeight = i2;
                            LogUtil.e("mWidth:" + PlayVideoActivity.this.mWidth + ", mHeight:" + PlayVideoActivity.this.mHeight);
                        }
                    });
                }
            });
        }
    }

    /* renamed from: com.hismith.hismithapp.PlayVideoActivity$4, reason: invalid class name */
    class AnonymousClass4 implements Player.Listener {
        AnonymousClass4() {
        }

        @Override // androidx.media3.common.Player.Listener
        public void onIsPlayingChanged(boolean z) {
            if (z) {
                Log.w("Player", "真正开始播放视频/音频");
                PlayVideoActivity.this.binding.imgVideoSelect.setVisibility(8);
                PlayVideoActivity.this.binding.imgVideoPanel.setVisibility(8);
                if (PlayVideoActivity.this.binding.imgPanelGesture.getVisibility() == 0) {
                    PlayVideoActivity.this.binding.imgPanelGesture.clearAnimation();
                    PlayVideoActivity.this.binding.imgPanelGesture.setVisibility(8);
                }
                PlayVideoActivity.this.binding.imgVideoPlay.setImageResource(R.mipmap.ic_video_pause);
                PlayVideoActivity.this.mHandler.sendEmptyMessage(PlayVideoActivity.WHAT_PLAY_TIME);
                return;
            }
            Log.w("Player", "播放停止（可能是因为暂停、缓冲结束或错误）");
            PlayVideoActivity.this.binding.imgVideoSelect.setVisibility(0);
            PlayVideoActivity.this.binding.imgVideoPanel.setVisibility(0);
            ConstraintLayout constraintLayout = PlayVideoActivity.this.binding.layoutSpeed;
            final PlayVideoActivity playVideoActivity = PlayVideoActivity.this;
            constraintLayout.post(new Runnable() { // from class: com.hismith.hismithapp.PlayVideoActivity$4$$ExternalSyntheticLambda0
                @Override // java.lang.Runnable
                public final void run() {
                    PlayVideoActivity.this.moveSpeedAwayFromVisibleIcons();
                }
            });
            PlayVideoActivity.this.mHandler.removeMessages(PlayVideoActivity.WHAT_PLAY_TIME);
            PlayVideoActivity.this.mSpeed = 0;
            PlayVideoActivity.this.mVibrate = 0;
            BleManager.getInstance(PlayVideoActivity.this).writeSpeed(0);
            if (ProductUtil.isSMVType()) {
                BleManager.getInstance(PlayVideoActivity.this).writeVibrate(0);
            }
        }

        @Override // androidx.media3.common.Player.Listener
        public void onPlayWhenReadyChanged(boolean z, int i) {
            if (z) {
                Log.w("Player", "用户要求播放，但可能仍在缓冲");
                PlayVideoActivity.this.binding.imgVideoPlay.setImageResource(R.mipmap.ic_video_pause);
            } else {
                Log.w("Player", "用户暂停播放");
                PlayVideoActivity.this.binding.imgVideoPlay.setImageResource(R.mipmap.ic_video_play);
            }
        }

        @Override // androidx.media3.common.Player.Listener
        public void onPlaybackStateChanged(int i) {
            if (i == 1) {
                Log.w("Player", "空闲");
                return;
            }
            if (i == 2) {
                Log.w("Player", "缓冲中...");
                PlayVideoActivity.this.binding.groupEmpty.setVisibility(8);
                PlayVideoActivity.this.binding.groupPlaying.setVisibility(0);
                return;
            }
            if (i == 3) {
                Log.w("Player", "准备好，可以播放");
                Log.w("Player", "getDuration:" + PlayVideoActivity.this.player.getDuration());
                long duration = PlayVideoActivity.this.player.getDuration();
                if (duration != C.TIME_UNSET) {
                    PlayVideoActivity.this.binding.seekVideo.setMax((int) duration);
                    PlayVideoActivity.this.binding.txtVideoTimeTotal.setText(" / " + ModeTimeUtil.formatMilliTime(duration));
                    return;
                }
                return;
            }
            if (i != 4) {
                return;
            }
            Log.w("Player", "播放结束");
            PlayVideoActivity.this.binding.imgVideoPlay.setImageResource(R.mipmap.ic_video_play);
            long duration2 = PlayVideoActivity.this.player.getDuration();
            if (duration2 != C.TIME_UNSET) {
                PlayVideoActivity.this.binding.seekVideo.setProgress((int) duration2);
                PlayVideoActivity.this.binding.txtVideoTimeCurrent.setText(ModeTimeUtil.formatMilliTime(duration2));
            }
        }

        @Override // androidx.media3.common.Player.Listener
        public void onPlayerError(PlaybackException playbackException) {
            Log.e("Player", "播放错误: " + playbackException.getMessage());
            PlayVideoActivity.this.mHandler.removeMessages(PlayVideoActivity.WHAT_PLAY_TIME);
            ToastUtil.showShortToast(PlayVideoActivity.this, LanguageUtil.getLanguageByKey(PlayVideoActivity.this, "failedToLoad"));
        }

        @Override // androidx.media3.common.Player.Listener
        public void onPositionDiscontinuity(Player.PositionInfo positionInfo, Player.PositionInfo positionInfo2, int i) {
            Log.w("Player", "位置跳跃 " + positionInfo2.positionMs + "，原因：" + i);
            if (i == 1 || i == 0) {
                PlayVideoActivity.this.moveSpeedTracker.reset();
                boolean seekToDetectVideoAsync = PlayVideoActivity.this.videoPoseLandmarkerHelper.seekToDetectVideoAsync(positionInfo2.positionMs);
                LogUtil.e("isBuffer:" + seekToDetectVideoAsync);
                if (PlayVideoActivity.this.player != null) {
                    PlayVideoActivity playVideoActivity = PlayVideoActivity.this;
                    if (seekToDetectVideoAsync) {
                        if (playVideoActivity.userPaused) {
                            return;
                        }
                        PlayVideoActivity.this.player.play();
                    } else {
                        playVideoActivity.binding.seekVideo.setSecondaryProgress(0);
                        PlayVideoActivity.this.player.pause();
                        PlayVideoActivity.this.showCachingHint();
                    }
                }
            }
        }

        @Override // androidx.media3.common.Player.Listener
        public void onVideoSizeChanged(final VideoSize videoSize) {
            Log.w("Player", "onVideoSizeChanged width:" + videoSize.width + ", height:" + videoSize.height);
            if (videoSize.width <= 0 || videoSize.height <= 0) {
                return;
            }
            PlayVideoActivity.this.binding.overlay.post(new Runnable() { // from class: com.hismith.hismithapp.PlayVideoActivity.4.1
                @Override // java.lang.Runnable
                public void run() {
                    if (PlayVideoActivity.this.binding.overlay.getWidth() == 0 || PlayVideoActivity.this.binding.overlay.getVisibility() != 0) {
                        return;
                    }
                    ViewGroup.LayoutParams layoutParams = PlayVideoActivity.this.binding.overlay.getLayoutParams();
                    layoutParams.height = (int) (PlayVideoActivity.this.binding.overlay.getWidth() / (videoSize.width / videoSize.height));
                    PlayVideoActivity.this.binding.overlay.setLayoutParams(layoutParams);
                }
            });
        }
    }

    private void addSpeedEdgeCandidates(ArrayList<float[]> arrayList, View view, View view2, float f, float f2) {
        if (view2.getVisibility() != 0) {
            return;
        }
        float left = view2.getLeft() + view2.getTranslationX();
        float right = view2.getRight() + view2.getTranslationX();
        float top = view2.getTop() + view2.getTranslationY();
        float bottom = view2.getBottom() + view2.getTranslationY();
        addSpeedTranslationCandidate(arrayList, (left - view.getRight()) - 1.0f, f2);
        addSpeedTranslationCandidate(arrayList, (right - view.getLeft()) + 1.0f, f2);
        addSpeedTranslationCandidate(arrayList, f, (top - view.getBottom()) - 1.0f);
        addSpeedTranslationCandidate(arrayList, f, (bottom - view.getTop()) + 1.0f);
    }

    private void addSpeedTranslationCandidate(ArrayList<float[]> arrayList, float f, float f2) {
        arrayList.add(new float[]{f, f2});
    }

    private void adjustPlayerViewHeight(final int i, final int i2) {
        if (i <= 0 || i2 <= 0) {
            return;
        }
        if (this.binding.playView.getWidth() == 0) {
            this.binding.playView.post(new Runnable() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda5
                @Override // java.lang.Runnable
                public final void run() {
                    PlayVideoActivity.this.m565x238335d7(i, i2);
                }
            });
        } else {
            m565x238335d7(i, i2);
        }
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void applyTrackCheck(int i) {
        this.moveSpeedTracker.setEnabledParts((i & 1) != 0, (i & 16) != 0, (i & 256) != 0);
    }

    /* JADX INFO: Access modifiers changed from: private */
    /* renamed from: calculateAndSetHeight, reason: merged with bridge method [inline-methods] */
    public void m565x238335d7(int i, int i2) {
        int width = this.binding.playView.getWidth();
        if (width == 0) {
            return;
        }
        int i3 = (int) ((width * i2) / i);
        LogUtil.e("calculateAndSetHeight getHeight:" + this.binding.playView.getHeight());
        ViewGroup.LayoutParams layoutParams = this.binding.playView.getLayoutParams();
        layoutParams.height = i3;
        this.binding.playView.setLayoutParams(layoutParams);
        LogUtil.e("calculateAndSetHeight height:" + i3);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public float[] clampSpeedTranslation(View view, float f, float f2) {
        float f3 = -view.getLeft();
        float width = this.binding.main.getWidth() - view.getRight();
        float bottom = this.binding.layoutTitle.getBottom() - view.getTop();
        float top = this.binding.layoutControl.getTop() - view.getBottom();
        float[] fArr = {Math.max(f3, Math.min(f, width)), Math.max(bottom, Math.min(f2, top))};
        keepSpeedAwayFromViews(view, fArr, f3, width, bottom, top);
        fArr[0] = Math.max(f3, Math.min(fArr[0], width));
        fArr[1] = Math.max(bottom, Math.min(fArr[1], top));
        return fArr;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void confirmExit() {
        FeatureGuideView featureGuideView = this.videoPanelTipView;
        if (featureGuideView != null && featureGuideView.getVisibility() == 0) {
            hideVideoPanelTip();
            return;
        }
        ExoPlayer exoPlayer = this.player;
        if (exoPlayer == null || exoPlayer.getMediaItemCount() <= 0) {
            finish();
        } else {
            DialogUtil.showMultipleConfirmDialog(this, "notice", "videoPlayingGoBack", null, null, "goBack", new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayVideoActivity.9
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    PlayVideoActivity.this.finish();
                }
            }, new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayVideoActivity.10
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    PlayVideoActivity.this.resetUI();
                    DialogUtil.dismissDialog();
                }
            });
        }
    }

    private float[] findNearestNonOverlappingTranslation(View view, float f, float f2, float f3, float f4, float f5, float f6) {
        ArrayList<float[]> arrayList = new ArrayList<>();
        addSpeedTranslationCandidate(arrayList, f, f2);
        addSpeedEdgeCandidates(arrayList, view, this.binding.imgVideoSelect, f, f2);
        addSpeedEdgeCandidates(arrayList, view, this.binding.imgVideoPanel, f, f2);
        Iterator<float[]> it = arrayList.iterator();
        float[] fArr = null;
        float f7 = Float.MAX_VALUE;
        while (it.hasNext()) {
            float[] next = it.next();
            float f8 = next[0];
            if (f8 >= f3 && f8 <= f4) {
                float f9 = next[1];
                if (f9 >= f5 && f9 <= f6 && getOverlappingSpeedObstacle(view, f8, f9) == null) {
                    float f10 = next[0] - f;
                    float f11 = next[1] - f2;
                    float f12 = (f10 * f10) + (f11 * f11);
                    if (f12 < f7) {
                        fArr = next;
                        f7 = f12;
                    }
                }
            }
        }
        return fArr;
    }

    private View getOverlappingSpeedObstacle(View view, float f, float f2) {
        boolean viewsOverlap = viewsOverlap(view, f, f2, this.binding.imgVideoSelect);
        ActivityPlayVideoBinding activityPlayVideoBinding = this.binding;
        if (viewsOverlap) {
            return activityPlayVideoBinding.imgVideoSelect;
        }
        if (viewsOverlap(view, f, f2, activityPlayVideoBinding.imgVideoPanel)) {
            return this.binding.imgVideoPanel;
        }
        return null;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void hideCachingHint() {
        this.isCaching = false;
        ActivityPlayVideoBinding activityPlayVideoBinding = this.binding;
        if (activityPlayVideoBinding != null) {
            activityPlayVideoBinding.layoutCachingHint.setVisibility(8);
        }
    }

    private void hideVideoPanelTip() {
        this.videoPanelTipView.setVisibility(8);
        this.binding.imgPanelGesture.setVisibility(0);
        AnimUtil.translate(this, this.binding.imgPanelGesture);
        this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda3
            @Override // java.lang.Runnable
            public final void run() {
                PlayVideoActivity.this.m566x48d2d872();
            }
        }, 5000L);
    }

    private void init() {
        LanguageUtil.setText(this, this.binding.titleTxtName, "playVideo");
        LanguageUtil.setText(this, this.binding.txtSelectVideo, "addVideo");
        LanguageUtil.setText(this, this.binding.txtVideoEmpty, "addVideoToPlay");
        LanguageUtil.setText(this, this.binding.txtCachingHint, "cachingPleaseWait");
        this.safeFuns1 = ((Integer) SharedPreferencesUtil.get(this, Constans.SAFE_FUN1, 40)).intValue();
        this.safeFuns2 = ((Integer) SharedPreferencesUtil.get(this, Constans.SAFE_FUN2, 40)).intValue();
        int status = BleManager.getInstance(this).getStatus();
        ActivityPlayVideoBinding activityPlayVideoBinding = this.binding;
        if (status == 2) {
            activityPlayVideoBinding.imgBluetooth.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_bluetooth_connect));
        } else {
            activityPlayVideoBinding.imgBluetooth.setImageDrawable(ContextCompat.getDrawable(this, R.mipmap.ic_play_bluetooth_disconnect));
        }
        if (SharedPreferencesUtil.contains(this, Constans.VIDEO_TRACK_CHECK)) {
            applyTrackCheck(((Integer) SharedPreferencesUtil.get(this, Constans.VIDEO_TRACK_CHECK, 273)).intValue());
        }
        this.videoPoseLandmarkerHelper = new PoseLandmarkerHelper5(this);
        ExoPlayer build = new ExoPlayer.Builder(this).build();
        this.player = build;
        build.setRepeatMode(1);
        this.binding.imgVideoLoop.setActivated(true);
        this.binding.playView.setPlayer(this.player);
        this.picker = new MediaPickerHelper(this, new AnonymousClass3());
        makeFilter();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initBleAAData(String str, String str2) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        str.hashCode();
        if (str.equals("03") || str.equals("04")) {
            try {
                int intValue = Integer.valueOf(str2, 16).intValue();
                if (intValue > 100) {
                    intValue = 100;
                } else if (intValue < 0) {
                    intValue = 0;
                }
                Message message = new Message();
                message.what = 3;
                message.obj = Integer.valueOf(intValue);
                this.mHandler.sendMessage(message);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void initVideoPanelTip() {
        FeatureGuideView featureGuideView = new FeatureGuideView(this);
        this.videoPanelTipView = featureGuideView;
        featureGuideView.setId(View.generateViewId());
        this.videoPanelTipView.setVisibility(8);
        this.binding.main.addView(this.videoPanelTipView, new ConstraintLayout.LayoutParams(-1, -1));
        this.videoPanelTipView.setTargetView(this.binding.imgVideoPanel);
        this.videoPanelTipView.setOnDismissedListener(new FeatureGuideView.OnDismissedListener() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda0
            @Override // com.hismith.hismithapp.view.FeatureGuideView.OnDismissedListener
            public final void onDismissed(boolean z) {
                PlayVideoActivity.this.m567xaf679745(z);
            }
        });
    }

    private void keepSpeedAwayFromViews(View view, float[] fArr, float f, float f2, float f3, float f4) {
        float[] findNearestNonOverlappingTranslation = findNearestNonOverlappingTranslation(view, fArr[0], fArr[1], f, f2, f3, f4);
        if (findNearestNonOverlappingTranslation != null) {
            fArr[0] = findNearestNonOverlappingTranslation[0];
            fArr[1] = findNearestNonOverlappingTranslation[1];
        }
    }

    static /* synthetic */ WindowInsetsCompat lambda$onCreate$0(View view, WindowInsetsCompat windowInsetsCompat) {
        Insets insets = windowInsetsCompat.getInsets(WindowInsetsCompat.Type.systemBars());
        view.setPadding(insets.left, insets.top, insets.right, insets.bottom);
        return windowInsetsCompat;
    }

    private void makeFilter() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_BLE_CONN_STATE);
        intentFilter.addAction(Constans.RECEIVER_PRODUCT_INFO);
        intentFilter.addAction(Constans.RECEIVER_BLE_RECE_DATA);
        AppUtil.registerReceiver(this, this.mReceiver, intentFilter);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void maybeShowVideoPanelTip() {
        if (this.videoPanelTipShown) {
            return;
        }
        this.videoPanelTipShown = true;
        if (((Boolean) SharedPreferencesUtil.get(this, Constans.VIDEO_PANEL_TIP_DISMISSED, false)).booleanValue()) {
            return;
        }
        this.videoPanelTipView.setVisibility(0);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void moveSpeedAwayFromVisibleIcons() {
        if (this.binding.layoutSpeed.getVisibility() != 0) {
            return;
        }
        float[] clampSpeedTranslation = clampSpeedTranslation(this.binding.layoutSpeed, this.binding.layoutSpeed.getTranslationX(), this.binding.layoutSpeed.getTranslationY());
        this.binding.layoutSpeed.setTranslationX(clampSpeedTranslation[0]);
        this.binding.layoutSpeed.setTranslationY(clampSpeedTranslation[1]);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void onSpeedUI(int i) {
        this.binding.txtSpeed.setText(i + "");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void resetUI() {
        ExoPlayer exoPlayer = this.player;
        if (exoPlayer != null) {
            exoPlayer.stop();
            this.player.clearMediaItems();
        }
        this.binding.seekVideo.setProgress(0);
        this.binding.seekVideo.setSecondaryProgress(0);
        this.binding.txtVideoTimeCurrent.setText(ModeTimeUtil.formatMilliTime(0));
        this.binding.txtVideoTimeTotal.setText(" / " + ModeTimeUtil.formatMilliTime(0));
        this.binding.overlay.clear();
        this.binding.imgVideoPlay.setImageResource(R.mipmap.ic_video_play);
        this.moveSpeedTracker.reset();
        this.lastMainPersonIndex = -1;
        this.lastMainFingerprint = null;
        this.userPaused = true;
        this.isCaching = false;
        hideCachingHint();
        this.binding.imgPanelGesture.clearAnimation();
        this.binding.imgPanelGesture.setVisibility(8);
        this.binding.groupEmpty.setVisibility(0);
        this.binding.groupPlaying.setVisibility(8);
        this.videoPoseLandmarkerHelper.cancelAll();
        this.mHandler.removeCallbacksAndMessages(null);
        BleManager.getInstance(this).writeShutdown();
    }

    private void runVideoLoop() {
        ExoPlayer exoPlayer = this.player;
        if (exoPlayer == null) {
            return;
        }
        int repeatMode = exoPlayer.getRepeatMode();
        ExoPlayer exoPlayer2 = this.player;
        if (repeatMode == 1) {
            exoPlayer2.setRepeatMode(0);
            this.binding.imgVideoLoop.setActivated(false);
        } else {
            exoPlayer2.setRepeatMode(1);
            this.binding.imgVideoLoop.setActivated(true);
        }
    }

    private void runVideoNext() {
        ExoPlayer exoPlayer = this.player;
        if (exoPlayer == null) {
            return;
        }
        long duration = exoPlayer.getDuration();
        if (duration <= 0) {
            return;
        }
        long min = Math.min(this.player.getCurrentPosition() + 15000, duration);
        this.player.seekTo(min);
        this.binding.txtVideoTimeCurrent.setText(ModeTimeUtil.formatMilliTime(min));
    }

    private void runVideoPrevious() {
        ExoPlayer exoPlayer = this.player;
        if (exoPlayer == null) {
            return;
        }
        long max = Math.max(exoPlayer.getCurrentPosition() - 5000, 0L);
        this.player.seekTo(max);
        this.binding.txtVideoTimeCurrent.setText(ModeTimeUtil.formatMilliTime(max));
    }

    /* JADX INFO: Access modifiers changed from: private */
    /* JADX WARN: Removed duplicated region for block: B:44:0x00de  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public void runVideoResult(long j) {
        List<NormalizedLandmark> list;
        int i;
        int min;
        int i2;
        int i3;
        int i4;
        int i5;
        if (!this.mIsSeeking) {
            this.binding.seekVideo.setProgress((int) j);
        }
        this.binding.txtVideoTimeCurrent.setText(ModeTimeUtil.formatMilliTime(j));
        PoseLandmarkerResult cacheResult = this.videoPoseLandmarkerHelper.getCacheResult(j);
        LogUtil.e("PoseLandmarkerResult result:" + (cacheResult != null ? cacheResult.toString() : "null"));
        char c = 0;
        if (cacheResult == null || cacheResult.landmarks() == null || cacheResult.landmarks().isEmpty()) {
            list = null;
            i = -1;
        } else {
            int selectMainPersonIndex = PoseMoveSpeedTracker.selectMainPersonIndex(cacheResult.landmarks());
            if (selectMainPersonIndex < 0 || (list = cacheResult.landmarks().get(selectMainPersonIndex)) == null || list.size() < 33) {
                list = null;
                i = selectMainPersonIndex;
            } else {
                int i6 = this.lastMainPersonIndex;
                if (i6 < 0 || i6 >= cacheResult.landmarks().size() || this.lastMainFingerprint == null) {
                    i3 = selectMainPersonIndex;
                } else {
                    float f = 0.2f;
                    int i7 = -1;
                    int i8 = 0;
                    while (i8 < cacheResult.landmarks().size()) {
                        float[] personFingerprint = PoseMoveSpeedTracker.personFingerprint(cacheResult.landmarks().get(i8));
                        if (personFingerprint == null) {
                            i5 = selectMainPersonIndex;
                        } else {
                            float f2 = personFingerprint[c];
                            float[] fArr = this.lastMainFingerprint;
                            i5 = selectMainPersonIndex;
                            float hypot = (float) Math.hypot(f2 - fArr[c], personFingerprint[1] - fArr[1]);
                            if (hypot < f) {
                                f = hypot;
                                i7 = i8;
                            }
                        }
                        i8++;
                        selectMainPersonIndex = i5;
                        c = 0;
                    }
                    i3 = selectMainPersonIndex;
                    if (i7 >= 0) {
                        list = cacheResult.landmarks().get(i7);
                        i4 = i7;
                        if (i4 != this.lastMainPersonIndex) {
                            this.moveSpeedTracker.reset();
                        }
                        this.lastMainPersonIndex = i4;
                        this.lastMainFingerprint = PoseMoveSpeedTracker.personFingerprint(list);
                        i = i4;
                    }
                }
                i4 = i3;
                if (i4 != this.lastMainPersonIndex) {
                }
                this.lastMainPersonIndex = i4;
                this.lastMainFingerprint = PoseMoveSpeedTracker.personFingerprint(list);
                i = i4;
            }
        }
        if (i < 0) {
            this.lastMainPersonIndex = -1;
            this.lastMainFingerprint = null;
        }
        int update = this.moveSpeedTracker.update(list, j);
        int min2 = Math.min(update, this.safeFuns1);
        int i9 = this.mSpeed;
        if (i9 != min2) {
            if (this.funs1Control) {
                this.mSpeed = min2;
                BleManager.getInstance(this).writeSpeed(min2);
            } else if (i9 != 0) {
                this.mSpeed = 0;
                BleManager.getInstance(this).writeSpeed(0);
            }
        }
        if (ProductUtil.isSMVType() && (i2 = this.mVibrate) != (min = Math.min(update, this.safeFuns2))) {
            if (this.funs2Control) {
                this.mVibrate = min;
                BleManager.getInstance(this).writeVibrate(min);
            } else if (i2 != 0) {
                this.mVibrate = 0;
                BleManager.getInstance(this).writeVibrate(0);
            }
        }
        if (list == null || this.binding.overlay.getVisibility() != 0) {
            this.binding.overlay.clear();
        } else {
            this.binding.overlay.setActivePart(this.moveSpeedTracker.getCurrentPart());
            this.binding.overlay.setResults(cacheResult, this.mHeight, this.mWidth, RunningMode.VIDEO, i);
        }
    }

    private void setListener() {
        this.player.addListener(new AnonymousClass4());
        this.binding.titleImgBack.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda6
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoActivity.this.m568lambda$setListener$1$comhismithhismithappPlayVideoActivity(view);
            }
        });
        this.binding.imgBluetooth.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda7
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoActivity.this.m569lambda$setListener$2$comhismithhismithappPlayVideoActivity(view);
            }
        });
        this.binding.imgSelectVideo.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda8
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoActivity.this.m570lambda$setListener$3$comhismithhismithappPlayVideoActivity(view);
            }
        });
        this.binding.imgVideoSelect.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda9
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoActivity.this.m571lambda$setListener$4$comhismithhismithappPlayVideoActivity(view);
            }
        });
        this.binding.imgVideoPlay.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda10
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoActivity.this.m572lambda$setListener$5$comhismithhismithappPlayVideoActivity(view);
            }
        });
        this.binding.imgVideoPrevious.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda11
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoActivity.this.m573lambda$setListener$6$comhismithhismithappPlayVideoActivity(view);
            }
        });
        this.binding.imgVideoNext.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda12
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoActivity.this.m574lambda$setListener$7$comhismithhismithappPlayVideoActivity(view);
            }
        });
        this.binding.imgVideoLoop.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda1
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoActivity.this.m575lambda$setListener$8$comhismithhismithappPlayVideoActivity(view);
            }
        });
        this.binding.seekVideo.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() { // from class: com.hismith.hismithapp.PlayVideoActivity.5
            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onProgressChanged(SeekBar seekBar, int i, boolean z) {
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStartTrackingTouch(SeekBar seekBar) {
                PlayVideoActivity.this.mIsSeeking = true;
            }

            @Override // android.widget.SeekBar.OnSeekBarChangeListener
            public void onStopTrackingTouch(SeekBar seekBar) {
                PlayVideoActivity.this.mIsSeeking = false;
                if (PlayVideoActivity.this.player != null) {
                    PlayVideoActivity.this.player.seekTo(seekBar.getProgress());
                    PlayVideoActivity.this.binding.txtVideoTimeCurrent.setText(ModeTimeUtil.formatMilliTime(seekBar.getProgress()));
                }
            }
        });
        this.binding.imgVideoPanel.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda2
            @Override // android.view.View.OnClickListener
            public final void onClick(View view) {
                PlayVideoActivity.this.m576lambda$setListener$9$comhismithhismithappPlayVideoActivity(view);
            }
        });
        this.binding.layoutSpeed.setOnTouchListener(new View.OnTouchListener() { // from class: com.hismith.hismithapp.PlayVideoActivity.6
            private float lastRawX;
            private float lastRawY;

            @Override // android.view.View.OnTouchListener
            public boolean onTouch(View view, MotionEvent motionEvent) {
                int actionMasked = motionEvent.getActionMasked();
                if (actionMasked == 0) {
                    this.lastRawX = motionEvent.getRawX();
                    this.lastRawY = motionEvent.getRawY();
                    return true;
                }
                if (actionMasked == 1) {
                    view.performClick();
                    return true;
                }
                if (actionMasked != 2) {
                    return true;
                }
                float rawX = motionEvent.getRawX() - this.lastRawX;
                float rawY = motionEvent.getRawY() - this.lastRawY;
                float[] clampSpeedTranslation = PlayVideoActivity.this.clampSpeedTranslation(view, view.getTranslationX() + rawX, view.getTranslationY() + rawY);
                view.setTranslationX(clampSpeedTranslation[0]);
                view.setTranslationY(clampSpeedTranslation[1]);
                this.lastRawX = motionEvent.getRawX();
                this.lastRawY = motionEvent.getRawY();
                return true;
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void showCachingHint() {
        this.isCaching = true;
        ActivityPlayVideoBinding activityPlayVideoBinding = this.binding;
        if (activityPlayVideoBinding != null) {
            activityPlayVideoBinding.layoutCachingHint.setVisibility(0);
        }
    }

    private void showPanelDialog() {
        this.binding.imgVideoPanel.setImageResource(R.mipmap.ic_video_panel_open);
        PlayVideoPanelDialog playVideoPanelDialog = new PlayVideoPanelDialog(this, this.safeFuns1, this.safeFuns2, this.funs1Control, this.funs2Control);
        this.mPanelDialog = playVideoPanelDialog;
        playVideoPanelDialog.setOnSafeChangeListener(new PlayVideoPanelDialog.onSafeChangeListener() { // from class: com.hismith.hismithapp.PlayVideoActivity.7
            @Override // com.hismith.hismithapp.dialog.PlayVideoPanelDialog.onSafeChangeListener
            public void onFuns2Control(boolean z) {
                PlayVideoActivity.this.funs2Control = z;
                if (PlayVideoActivity.this.funs2Control) {
                    return;
                }
                PlayVideoActivity.this.mVibrate = 0;
                BleManager.getInstance(PlayVideoActivity.this).writeVibrate(0);
            }

            @Override // com.hismith.hismithapp.dialog.PlayVideoPanelDialog.onSafeChangeListener
            public void onFuns2Safe(int i) {
                PlayVideoActivity.this.safeFuns2 = i;
                PlayVideoActivity playVideoActivity = PlayVideoActivity.this;
                SharedPreferencesUtil.put(playVideoActivity, Constans.SAFE_FUN2, Integer.valueOf(playVideoActivity.safeFuns2));
            }

            @Override // com.hismith.hismithapp.dialog.PlayVideoPanelDialog.onSafeChangeListener
            public void onSpeedControl(boolean z) {
                PlayVideoActivity.this.funs1Control = z;
                if (PlayVideoActivity.this.funs1Control) {
                    return;
                }
                PlayVideoActivity.this.mSpeed = 0;
                BleManager.getInstance(PlayVideoActivity.this).writeSpeed(0);
            }

            @Override // com.hismith.hismithapp.dialog.PlayVideoPanelDialog.onSafeChangeListener
            public void onSpeedSafe(int i) {
                PlayVideoActivity.this.safeFuns1 = i;
                PlayVideoActivity playVideoActivity = PlayVideoActivity.this;
                SharedPreferencesUtil.put(playVideoActivity, Constans.SAFE_FUN1, Integer.valueOf(playVideoActivity.safeFuns1));
            }

            @Override // com.hismith.hismithapp.dialog.PlayVideoPanelDialog.onSafeChangeListener
            public void onTrackCheckChange(int i) {
                PlayVideoActivity.this.applyTrackCheck(i);
            }
        });
        this.mPanelDialog.setOnDialogStatusListener(new PlayVideoPanelDialog.onDialogStatusListener() { // from class: com.hismith.hismithapp.PlayVideoActivity.8
            @Override // com.hismith.hismithapp.dialog.PlayVideoPanelDialog.onDialogStatusListener
            public void beforeDismiss() {
                PlayVideoActivity.this.binding.imgVideoPanel.setImageResource(R.mipmap.ic_video_panel_close);
                PlayVideoActivity.this.restore();
            }

            @Override // com.hismith.hismithapp.dialog.PlayVideoPanelDialog.onDialogStatusListener
            public void beforeShow(int i) {
                ConstraintSet constraintSet = new ConstraintSet();
                constraintSet.clone(PlayVideoActivity.this.binding.main);
                ConstraintSet constraintSet2 = new ConstraintSet();
                constraintSet2.clone(constraintSet);
                constraintSet2.clear(R.id.playView, 3);
                constraintSet2.clear(R.id.playView, 4);
                constraintSet2.connect(R.id.playView, 3, R.id.layout_title, 4);
                constraintSet2.connect(R.id.playView, 4, R.id.view_height, 3);
                constraintSet2.constrainHeight(R.id.view_height, i);
                constraintSet2.connect(R.id.view_height, 4, 0, 4);
                AutoTransition autoTransition = new AutoTransition();
                autoTransition.setDuration(PlayVideoActivity.this.animDuration);
                TransitionManager.beginDelayedTransition(PlayVideoActivity.this.binding.main, autoTransition);
                constraintSet2.applyTo(PlayVideoActivity.this.binding.main);
            }
        });
        this.mPanelDialog.showDialog();
    }

    private void unRegisterReceiver() {
        try {
            unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private boolean viewsOverlap(View view, float f, float f2, View view2) {
        if (view2.getVisibility() != 0) {
            return false;
        }
        float left = view.getLeft() + f;
        float right = view.getRight() + f;
        float top = view.getTop() + f2;
        float bottom = view.getBottom() + f2;
        return left < ((float) view2.getRight()) + view2.getTranslationX() && right > view2.getLeft() + view2.getTranslationX() && top < ((float) view2.getBottom()) + view2.getTranslationY() && bottom > ((float) view2.getTop()) + view2.getTranslationY();
    }

    public void initBleCCData(String str, String str2, String str3) {
        if (TextUtils.isEmpty(str2)) {
            return;
        }
        try {
            int hashCode = str2.hashCode();
            if (hashCode == 1539) {
                if (str2.equals("03")) {
                    Message message = new Message();
                    message.what = 3;
                    message.obj = Integer.valueOf(str3, 16);
                    Bundle bundle = new Bundle();
                    bundle.putString(Constans.BleType, str);
                    message.setData(bundle);
                    this.mHandler.sendMessage(message);
                    return;
                }
                return;
            }
            if (hashCode == 1541 && str2.equals("05")) {
                if ((!this.funs1Control || this.safeFuns1 == 0) && this.funs2Control) {
                    Message message2 = new Message();
                    message2.what = 3;
                    message2.obj = Integer.valueOf(str3, 16);
                    Bundle bundle2 = new Bundle();
                    bundle2.putString(Constans.BleType, str);
                    message2.setData(bundle2);
                    this.mHandler.sendMessage(message2);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* renamed from: lambda$hideVideoPanelTip$11$com-hismith-hismithapp-PlayVideoActivity, reason: not valid java name */
    /* synthetic */ void m566x48d2d872() {
        this.binding.imgPanelGesture.clearAnimation();
        this.binding.imgPanelGesture.setVisibility(8);
    }

    /* renamed from: lambda$initVideoPanelTip$10$com-hismith-hismithapp-PlayVideoActivity, reason: not valid java name */
    /* synthetic */ void m567xaf679745(boolean z) {
        if (z) {
            SharedPreferencesUtil.put(this, Constans.VIDEO_PANEL_TIP_DISMISSED, true);
        }
        hideVideoPanelTip();
    }

    /* renamed from: lambda$setListener$1$com-hismith-hismithapp-PlayVideoActivity, reason: not valid java name */
    /* synthetic */ void m568lambda$setListener$1$comhismithhismithappPlayVideoActivity(View view) {
        confirmExit();
    }

    /* renamed from: lambda$setListener$2$com-hismith-hismithapp-PlayVideoActivity, reason: not valid java name */
    /* synthetic */ void m569lambda$setListener$2$comhismithhismithappPlayVideoActivity(View view) {
        this.mIsEnterBluetooth = true;
        startActivity(new Intent(this, (Class<?>) BluetootListActivity.class));
    }

    /* renamed from: lambda$setListener$3$com-hismith-hismithapp-PlayVideoActivity, reason: not valid java name */
    /* synthetic */ void m570lambda$setListener$3$comhismithhismithappPlayVideoActivity(View view) {
        this.picker.pickVideo();
    }

    /* renamed from: lambda$setListener$4$com-hismith-hismithapp-PlayVideoActivity, reason: not valid java name */
    /* synthetic */ void m571lambda$setListener$4$comhismithhismithappPlayVideoActivity(View view) {
        this.picker.pickVideo();
    }

    /* renamed from: lambda$setListener$5$com-hismith-hismithapp-PlayVideoActivity, reason: not valid java name */
    /* synthetic */ void m572lambda$setListener$5$comhismithhismithappPlayVideoActivity(View view) {
        if (this.isCaching) {
            ToastUtil.showShortToast(this, LanguageUtil.getLanguageByKey(this, "cachingPleaseWait"));
            return;
        }
        int playbackState = this.player.getPlaybackState();
        ExoPlayer exoPlayer = this.player;
        if (exoPlayer != null && !exoPlayer.isPlaying()) {
            if (playbackState == 4) {
                this.player.seekTo(0L);
            }
            this.userPaused = false;
            this.player.play();
            return;
        }
        ExoPlayer exoPlayer2 = this.player;
        if (exoPlayer2 == null || !exoPlayer2.isPlaying()) {
            return;
        }
        this.userPaused = true;
        this.player.pause();
    }

    /* renamed from: lambda$setListener$6$com-hismith-hismithapp-PlayVideoActivity, reason: not valid java name */
    /* synthetic */ void m573lambda$setListener$6$comhismithhismithappPlayVideoActivity(View view) {
        runVideoPrevious();
    }

    /* renamed from: lambda$setListener$7$com-hismith-hismithapp-PlayVideoActivity, reason: not valid java name */
    /* synthetic */ void m574lambda$setListener$7$comhismithhismithappPlayVideoActivity(View view) {
        runVideoNext();
    }

    /* renamed from: lambda$setListener$8$com-hismith-hismithapp-PlayVideoActivity, reason: not valid java name */
    /* synthetic */ void m575lambda$setListener$8$comhismithhismithappPlayVideoActivity(View view) {
        runVideoLoop();
    }

    /* renamed from: lambda$setListener$9$com-hismith-hismithapp-PlayVideoActivity, reason: not valid java name */
    /* synthetic */ void m576lambda$setListener$9$comhismithhismithappPlayVideoActivity(View view) {
        PlayVideoPanelDialog playVideoPanelDialog = this.mPanelDialog;
        if (playVideoPanelDialog == null || !playVideoPanelDialog.isShow()) {
            showPanelDialog();
        } else {
            this.mPanelDialog.dissDialog();
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        getWindow().addFlags(128);
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) { // from class: com.hismith.hismithapp.PlayVideoActivity.2
            @Override // androidx.activity.OnBackPressedCallback
            public void handleOnBackPressed() {
                PlayVideoActivity.this.confirmExit();
            }
        });
        ActivityPlayVideoBinding inflate = ActivityPlayVideoBinding.inflate(getLayoutInflater());
        this.binding = inflate;
        setContentView(inflate.getRoot());
        LogUtil.e("PlayVideoActivity onCreate");
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), new OnApplyWindowInsetsListener() { // from class: com.hismith.hismithapp.PlayVideoActivity$$ExternalSyntheticLambda4
            @Override // androidx.core.view.OnApplyWindowInsetsListener
            public final WindowInsetsCompat onApplyWindowInsets(View view, WindowInsetsCompat windowInsetsCompat) {
                return PlayVideoActivity.lambda$onCreate$0(view, windowInsetsCompat);
            }
        });
        initVideoPanelTip();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
        this.videoPoseLandmarkerHelper.cancelAll();
        this.mHandler.removeCallbacksAndMessages(null);
        BleManager.getInstance(this).writeShutdown();
        ExoPlayer exoPlayer = this.player;
        if (exoPlayer != null) {
            exoPlayer.release();
            this.player = null;
        }
        unRegisterReceiver();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onResume() {
        super.onResume();
        this.mIsEnterBluetooth = false;
    }

    @Override // androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onStop() {
        super.onStop();
        LogUtil.e("PlayVideoActivity mIsEnterBluetooth:" + this.mIsEnterBluetooth);
        if (this.mIsEnterBluetooth) {
            return;
        }
        ExoPlayer exoPlayer = this.player;
        if (exoPlayer != null && exoPlayer.isPlaying()) {
            this.player.pause();
        }
        this.userPaused = true;
        this.mHandler.removeMessages(WHAT_PLAY_TIME);
        this.mSpeed = 0;
        this.mVibrate = 0;
        BleManager.getInstance(this).writeSpeed(0);
        if (ProductUtil.isSMVType()) {
            BleManager.getInstance(this).writeVibrate(0);
        }
    }

    public void restore() {
        ConstraintSet constraintSet = new ConstraintSet();
        constraintSet.clone(this.binding.main);
        ConstraintSet constraintSet2 = new ConstraintSet();
        constraintSet2.clone(constraintSet);
        constraintSet2.clear(R.id.playView, 3);
        constraintSet2.clear(R.id.playView, 4);
        constraintSet2.connect(R.id.playView, 3, 0, 3);
        constraintSet2.connect(R.id.playView, 4, 0, 4);
        constraintSet2.constrainHeight(R.id.view_height, 0);
        constraintSet2.clear(R.id.view_height, 3);
        constraintSet2.connect(R.id.view_height, 4, 0, 4);
        AutoTransition autoTransition = new AutoTransition();
        autoTransition.setDuration(this.animDuration);
        TransitionManager.beginDelayedTransition(this.binding.main, autoTransition);
        constraintSet2.applyTo(this.binding.main);
    }
}
