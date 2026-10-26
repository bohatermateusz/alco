package com.hismith.hismithapp.surface_view;

import android.animation.Animator;
import android.animation.ValueAnimator;
import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Region;
import android.graphics.Shader;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.util.AttributeSet;
import android.view.Choreographer;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.animation.LinearInterpolator;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.surface_view.listener.OnPlayTimeListener;
import com.hismith.hismithapp.surface_view.listener.OnRecodeTimeListener;
import com.hismith.hismithapp.surface_view.listener.OnSelectTimeListener;
import com.hismith.hismithapp.surface_view.listener.OnStatusChangeListener;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/* loaded from: classes3.dex */
public class LineDrawView5 extends SurfaceView implements SurfaceHolder.Callback, Runnable {
    private static final int PIU_TOUCH_DEATH = 3;
    private static final int PIU_TOUCH_DRAG = 1;
    private static final int PIU_TOUCH_NONE = 0;
    private static final int PIU_TOUCH_SETTLE = 2;
    private final int PIU_COLOR;
    private final int PIU_DEATH_DURATION;
    private float PIU_HIT_RADIUS;
    private final float PIU_MAX_DISTANCE;
    private final float PIU_ORIGIN_DOT_RATIO;
    private final int PIU_SETTLE_DURATION;
    private final int WHAT_COMPLETE;
    private final int WHAT_FUN_DATA;
    private final int WHAT_PAUSE;
    private final int WHAT_PLAY_TIME;
    private final int WHAT_POSITION_RANGE;
    private final int WHAT_RECODE_TIME;
    private final int WHAT_SMOOTH;
    private final int WHAT_SPEED;
    private final int WHAT_STATUS;
    private final int WHAT_VIBRATE;
    private String[] YData;
    private int bgColor;
    private int cursorColor;
    private float distance;
    private float downOneX;
    private float downTwoX;
    private int dullColor;
    private int effectColor;
    private Map<Integer, String> funDataMap;
    private Map<Integer, String> funVibrateDataMap;
    private long high;
    private int inRectTarget;
    private int intervalTime;
    private boolean isDrawPiu;
    private boolean isDrawYScale;
    private boolean isEndSelectAnim;
    private float isEndSelectAnimMoveTime;
    private int isPause;
    private boolean isPiuDragRemoveEnabled;
    private boolean isPlay;
    private boolean isPlayCircle;
    private boolean isPrepare;
    private boolean isRecord;
    private boolean isRunning;
    private boolean isSelect;
    private boolean isSelectList;
    private boolean isSetData;
    private boolean isSmoothLine;
    private boolean isStartPositionLine;
    private boolean isStartSelectAnim;
    private float isStartSelectAnimMoveTime;
    private boolean isVibrateLine;
    private long lastEnd;
    private int lastSmooth;
    private int lastSpeed;
    private int lastStartPosition;
    private int lastVibrate;
    private Canvas mCanvas;
    private Paint[] mCirclePaints;
    private String mCopyData;
    private Map<Integer, String> mCopyFunDataMap;
    private String mCopySmoothData;
    private String mCopyStartPositionData;
    private String mCopyVibrateData;
    private int mCurrentScale;
    private int mCurrentVibrateScale;
    float mCursorBorderRadius;
    private Paint mCursorPaint;
    private float mDistance;
    private PiuItem mDragItem;
    private float mDragOriX;
    private float mDragOriY;
    private float mDragTouchOffsetX;
    private float mDragTouchOffsetY;
    private float mDragX;
    private float mDragY;
    private Paint mDullPaint;
    private Paint mEffectPaint;
    private Paint mFunPaint;
    private Bitmap mFunPiuIcon;
    private Handler mHandler;
    private int mHeight;
    private SurfaceHolder mHolder;
    private RectF mLeftRectF;
    private float mLineTextIntersticeWidth;
    private float mMaxScaleItemWidth;
    private float mMinScaleItemWidth;
    private float mOffX;
    private Paint[] mPaints;
    private Path[] mPaths;
    private Path mPiuAdhesionPath;
    private long mPiuAnimStartTime;
    private Paint mPiuDotPaint;
    private List<PiuItem> mPiuItems;
    private final Object mPiuLock;
    private Paint mPiuPiecePaint;
    private Paint mPiuSparkPaint;
    private int mPiuTouchState;
    private ValueAnimator mPlayAnimator;
    private Paint[] mPlayPaints;
    private Path[] mPlayPaths;
    private float mPlayPointTime;
    private int mPlayTime;
    private float mPlayX;
    private RectF mRightRectF;
    private int mScaleBottom;
    private int mScaleBottomY;
    private int mScaleHeight;
    private float mScaleItemWidth;
    private int mScaleLeft;
    private int mScaleLeftX;
    private int mScaleNum;
    private Paint mScalePaint;
    private int mScaleRight;
    private int mScaleRightX;
    private int mScaleShortNum;
    private int mScaleTop;
    private int mScaleTopY;
    private int mScaleWidth;
    private int mScreenTime;
    private int mSelectEndTime;
    private int mSelectLineWidth;
    private Paint mSelectPaint;
    private int mSelectStartTime;
    private String mSeparator;
    private float mSettleFromX;
    private float mSettleFromY;
    private int mShowTime;
    private int mSmoothScale;
    private int mStartPositionScale;
    private float mStrokeWidth;
    private int mTextHeight;
    private Paint mTextPaint;
    private float mTextSize;
    private Thread mThread;
    private float mThumbOffX;
    private int mThumbTime;
    private int mThumbWidth;
    private int mTime;
    private int mType;
    private int mWidth;
    private int mXScaleLongLine;
    private int mXScaleShortLine;
    private float mYHeight;
    private float mYItemHeight;
    private int mYItemNum;
    private int mYScaleLongLine;
    private int mYScaleShortLine;
    private float moveCenterX;
    private final int moveCursorMultiple;
    private boolean needDelay;
    private OnPlayTimeListener onPlayTimeListener;
    private OnRecodeTimeListener onRecodeTimeListener;
    private OnSelectTimeListener onSelectTimeListener;
    private OnStatusChangeListener onStatusChangeListener;
    private int selectColor;
    private StringBuffer strDataBuffer;
    private StringBuffer strSmoothDataBuffer;
    private StringBuffer strStartPositionDataBuffer;
    private StringBuffer strVibrateDataBuffer;
    private int textColor;
    int whiteHeight;
    int whiteWidth;

    private static class PiuItem {
        int actualTime;
        float x;
        float y;

        private PiuItem() {
        }
    }

    public LineDrawView5(Context context) {
        this(context, null);
    }

    public LineDrawView5(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public LineDrawView5(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.isPrepare = false;
        this.isPause = 0;
        this.isSetData = false;
        this.isPlay = false;
        this.isSelect = false;
        this.isSelectList = false;
        this.isPlayCircle = false;
        this.isDrawYScale = true;
        this.isDrawPiu = true;
        this.isPiuDragRemoveEnabled = true;
        this.mThumbWidth = MeasureUtil.dip2px(getContext(), 15.0f);
        this.mScaleTop = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mScaleBottom = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mScaleLeft = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mScaleRight = MeasureUtil.dip2px(getContext(), 1.0f);
        this.bgColor = 0;
        this.mStrokeWidth = MeasureUtil.dip2px(getContext(), 2.0f);
        this.mScaleNum = 0;
        this.mScaleShortNum = 2;
        this.mOffX = 0.0f;
        this.mThumbOffX = 0.0f;
        this.mXScaleLongLine = MeasureUtil.dip2px(getContext(), 10.0f);
        this.mXScaleShortLine = MeasureUtil.dip2px(getContext(), 5.0f);
        this.mYItemNum = 10;
        this.mTextSize = MeasureUtil.sp2px(getContext(), 12.0f);
        this.mLineTextIntersticeWidth = MeasureUtil.dip2px(getContext(), 5.0f);
        this.mYScaleLongLine = MeasureUtil.dip2px(getContext(), 12.0f);
        this.mYScaleShortLine = MeasureUtil.dip2px(getContext(), 6.0f);
        this.YData = new String[]{"100", "80", "60", "40", "20", "0"};
        this.mSelectStartTime = 0;
        this.mSelectEndTime = 0;
        this.moveCursorMultiple = 10;
        this.mTime = 0;
        this.mPlayTime = 0;
        this.mShowTime = 0;
        this.mPlayX = 0.0f;
        this.mPlayPointTime = 0.0f;
        this.mSeparator = "G";
        this.strDataBuffer = new StringBuffer("00" + this.mSeparator);
        this.strVibrateDataBuffer = new StringBuffer("00" + this.mSeparator);
        this.strStartPositionDataBuffer = new StringBuffer("00" + this.mSeparator);
        this.strSmoothDataBuffer = new StringBuffer("00" + this.mSeparator);
        this.mCopyData = "";
        this.mCopyVibrateData = "";
        this.mCopyStartPositionData = "";
        this.mCopySmoothData = "";
        this.mCopyFunDataMap = new LinkedHashMap();
        this.intervalTime = 25;
        this.mCurrentScale = 0;
        this.mCurrentVibrateScale = 0;
        this.mStartPositionScale = 0;
        this.mSmoothScale = 0;
        this.isVibrateLine = false;
        this.isStartPositionLine = false;
        this.isSmoothLine = false;
        this.mType = 1;
        this.PIU_MAX_DISTANCE = MeasureUtil.dip2px(getContext(), 60.0f);
        this.PIU_HIT_RADIUS = 0.0f;
        this.PIU_ORIGIN_DOT_RATIO = 0.5f;
        this.PIU_DEATH_DURATION = 350;
        this.PIU_SETTLE_DURATION = 200;
        this.PIU_COLOR = Color.parseColor("#FF00EA");
        this.mPiuLock = new Object();
        this.mPiuItems = new ArrayList();
        this.mPiuTouchState = 0;
        this.WHAT_RECODE_TIME = 1;
        this.WHAT_PLAY_TIME = 2;
        this.WHAT_STATUS = 3;
        this.WHAT_COMPLETE = 4;
        this.WHAT_SPEED = 5;
        this.WHAT_VIBRATE = 6;
        this.WHAT_PAUSE = 7;
        this.WHAT_FUN_DATA = 8;
        this.WHAT_POSITION_RANGE = 9;
        this.WHAT_SMOOTH = 16;
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.surface_view.LineDrawView5.2
            /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                int i2 = message.what;
                if (i2 != 16) {
                    switch (i2) {
                        case 1:
                            if (LineDrawView5.this.onRecodeTimeListener != null) {
                                LineDrawView5.this.onRecodeTimeListener.onRecodeTime(LineDrawView5.this.mTime);
                                break;
                            }
                            break;
                        case 2:
                            int intValue = ((Integer) message.obj).intValue();
                            if (LineDrawView5.this.onPlayTimeListener != null) {
                                LineDrawView5.this.onPlayTimeListener.onPlayTime(intValue);
                                break;
                            }
                            break;
                        case 3:
                            if (LineDrawView5.this.onStatusChangeListener != null) {
                                LineDrawView5.this.onStatusChangeListener.OnStatusChange();
                                break;
                            }
                            break;
                        case 4:
                            if (LineDrawView5.this.onPlayTimeListener != null) {
                                LineDrawView5.this.onPlayTimeListener.onComplete();
                                break;
                            }
                            break;
                        case 5:
                            int intValue2 = ((Integer) message.obj).intValue();
                            if (LineDrawView5.this.onPlayTimeListener != null) {
                                LineDrawView5.this.onPlayTimeListener.onSpeedChange(intValue2);
                                break;
                            }
                            break;
                        case 6:
                            int intValue3 = ((Integer) message.obj).intValue();
                            if (LineDrawView5.this.onPlayTimeListener != null) {
                                LineDrawView5.this.onPlayTimeListener.onVibrateChange(intValue3);
                                break;
                            }
                            break;
                        case 7:
                            if (LineDrawView5.this.onPlayTimeListener != null) {
                                LineDrawView5.this.onPlayTimeListener.onLinePause();
                                break;
                            }
                            break;
                        case 8:
                            String str = (String) message.obj;
                            if (!TextUtils.isEmpty(str) && str.length() > 2 && LineDrawView5.this.onPlayTimeListener != null) {
                                LineDrawView5.this.onPlayTimeListener.onFunData(str.substring(0, 2), str.substring(2));
                                break;
                            }
                            break;
                        case 9:
                            int i3 = message.arg1;
                            int i4 = message.arg2;
                            if (LineDrawView5.this.onPlayTimeListener != null) {
                                LineDrawView5.this.onPlayTimeListener.onPositionRangeChange(i3, i4);
                                break;
                            }
                            break;
                    }
                } else {
                    int intValue4 = ((Integer) message.obj).intValue();
                    if (LineDrawView5.this.onPlayTimeListener != null) {
                        LineDrawView5.this.onPlayTimeListener.onSmoothChange(intValue4);
                    }
                }
                return false;
            }
        });
        this.high = 0L;
        this.lastEnd = 0L;
        this.inRectTarget = 0;
        this.mSelectLineWidth = MeasureUtil.dip2px(getContext(), 15.0f);
        this.whiteWidth = MeasureUtil.dip2px(getContext(), 1.0f);
        this.whiteHeight = MeasureUtil.dip2px(getContext(), 20.0f);
        this.mCursorBorderRadius = MeasureUtil.dip2px(getContext(), 2.0f);
        initAttr(attributeSet);
        init();
    }

    private void buildAdhesionPath(float f, float f2, float f3, float f4, float f5, float f6) {
        float f7 = f4 - f;
        float f8 = f5 - f2;
        float hypot = (float) Math.hypot(f7, f8);
        if (hypot < 0.001f) {
            return;
        }
        float f9 = f7 / hypot;
        float f10 = -(f8 / hypot);
        float f11 = (f + f4) * 0.5f;
        float f12 = (f2 + f5) * 0.5f;
        float f13 = f10 * f3;
        float f14 = f3 * f9;
        float f15 = f2 + f14;
        float f16 = f2 - f14;
        float f17 = f10 * f6;
        float f18 = f9 * f6;
        this.mPiuAdhesionPath.moveTo(f + f13, f15);
        this.mPiuAdhesionPath.quadTo(f11, f12, f4 + f17, f5 + f18);
        this.mPiuAdhesionPath.lineTo(f4 - f17, f5 - f18);
        this.mPiuAdhesionPath.quadTo(f11, f12, f - f13, f16);
        this.mPiuAdhesionPath.close();
    }

    /* JADX WARN: Removed duplicated region for block: B:110:0x0145  */
    /* JADX WARN: Removed duplicated region for block: B:124:0x0183  */
    /* JADX WARN: Removed duplicated region for block: B:56:0x031e A[Catch: all -> 0x032d, Exception -> 0x0330, TRY_LEAVE, TryCatch #0 {Exception -> 0x0330, blocks: (B:23:0x01b3, B:25:0x01bd, B:27:0x01cd, B:28:0x0204, B:30:0x0208, B:31:0x0221, B:33:0x0225, B:34:0x023e, B:36:0x0261, B:38:0x0265, B:41:0x026a, B:43:0x026e, B:44:0x0277, B:46:0x027f, B:47:0x029c, B:49:0x02a0, B:50:0x02bd, B:52:0x02c1, B:53:0x02de, B:54:0x02fe, B:56:0x031e), top: B:22:0x01b3, outer: #2 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private void draw() {
        int i;
        int i2;
        int i3;
        int i4;
        int i5;
        Canvas canvas;
        SurfaceHolder surfaceHolder;
        char c;
        float f;
        char c2;
        int i6;
        int i7;
        int i8;
        int i9;
        float f2;
        float f3;
        int i10;
        boolean z = this.isRecord;
        if ((z || this.isPause == 1) && this.isRunning) {
            if (z) {
                if (this.mTime % 100 == 0) {
                    int i11 = this.mType;
                    if (i11 == 1) {
                        sendSpeed(this.mCurrentScale);
                        sendVibrate(this.mCurrentVibrateScale);
                    } else if (i11 == 2) {
                        sendSpeed(this.mCurrentScale);
                        sendPositionRange(this.mStartPositionScale, this.mCurrentVibrateScale);
                        sendSmooth(this.mSmoothScale);
                    }
                }
                this.mTime += this.intervalTime;
                setModeData(this.mCurrentScale);
                setVibrateModeData(this.mCurrentVibrateScale);
                setStartPositionModeData(this.mStartPositionScale);
                setSmoothModeData(this.mSmoothScale);
                this.mHandler.sendEmptyMessage(1);
            } else if (this.lastSpeed != 0 || this.lastVibrate != 0) {
                sendPause();
            }
            int i12 = this.mTime;
            int i13 = this.mScreenTime;
            if (i12 > i13) {
                i = i12 - i13;
                i2 = 0;
                i3 = 0;
                i4 = 0;
                i5 = 0;
            }
            i2 = 0;
            i3 = 0;
            i4 = 0;
            i5 = 0;
            i = 0;
        } else {
            if (this.isPlay || this.isPause == 2) {
                int i14 = this.mPlayTime / this.intervalTime;
                try {
                    int i15 = i14 * 3;
                    i3 = Integer.valueOf(this.strDataBuffer.substring(i15, i15 + 2), 16).intValue();
                } catch (Exception unused) {
                    i3 = 0;
                }
                try {
                    int i16 = i14 * 3;
                    i4 = Integer.valueOf(this.strVibrateDataBuffer.substring(i16, i16 + 2), 16).intValue();
                } catch (Exception unused2) {
                    i4 = 0;
                }
                try {
                    int i17 = i14 * 3;
                    i5 = Integer.valueOf(this.strStartPositionDataBuffer.substring(i17, i17 + 2), 16).intValue();
                } catch (Exception unused3) {
                    i5 = 0;
                }
                try {
                    int i18 = i14 * 3;
                    i2 = Integer.valueOf(this.strSmoothDataBuffer.substring(i18, i18 + 2), 16).intValue();
                } catch (Exception unused4) {
                    i2 = 0;
                }
                int i19 = this.mPlayTime;
                int i20 = this.mScreenTime;
                if (i19 < i20 / 2) {
                    float f4 = (i19 * 1.0f) / this.intervalTime;
                    this.mPlayPointTime = f4;
                    f2 = f4 * this.mDistance;
                    this.mPlayX = f2;
                } else {
                    int i21 = this.mTime;
                    if (i21 - i19 > i20 / 2) {
                        float f5 = (i20 / 2.0f) / this.intervalTime;
                        this.mPlayPointTime = f5;
                        f3 = f5 * this.mDistance;
                        this.mPlayX = f3;
                        i10 = i19 - (i20 / 2);
                    } else {
                        int i22 = this.intervalTime;
                        if (i21 > i20) {
                            float f6 = ((i20 - (i21 - i19)) * 1.0f) / i22;
                            this.mPlayPointTime = f6;
                            f2 = f6 * this.mDistance;
                            this.mPlayX = f2;
                        } else {
                            float f7 = (i19 * 1.0f) / i22;
                            this.mPlayPointTime = f7;
                            f2 = f7 * this.mDistance;
                            this.mPlayX = f2;
                        }
                        int i23 = i21 - i20;
                        if (i23 >= 0) {
                            f3 = f2;
                            i10 = i23;
                        }
                    }
                    this.mPlayX = f3 + this.mScaleLeftX;
                    sendPlayTime();
                    if (!this.isPlay) {
                        sendPlayFunData();
                        int i24 = this.mPlayTime;
                        if (i24 >= this.mTime) {
                            this.isPlay = false;
                            this.isPause = 2;
                            sendStatus();
                            sendComplete();
                        } else {
                            if (i24 % 100 < this.intervalTime) {
                                int i25 = this.mType;
                                if (i25 == 1) {
                                    sendSpeed(i3);
                                    sendVibrate(i4);
                                } else if (i25 == 2) {
                                    sendSpeed(i3);
                                    sendPositionRange(i5, i4);
                                    sendSmooth(i2);
                                }
                            }
                            int i26 = this.mPlayTime + this.intervalTime;
                            this.mPlayTime = i26;
                            int i27 = this.mTime;
                            if (i26 > i27) {
                                this.mPlayTime = i27;
                            }
                        }
                    } else if (this.lastSpeed != 0 || this.lastVibrate != 0) {
                        sendPause();
                    }
                    i = i10;
                }
                f3 = f2;
                i10 = 0;
                this.mPlayX = f3 + this.mScaleLeftX;
                sendPlayTime();
                if (!this.isPlay) {
                }
                i = i10;
            }
            i2 = 0;
            i3 = 0;
            i4 = 0;
            i5 = 0;
            i = 0;
        }
        this.mShowTime = i;
        if (this.isStartSelectAnim) {
            moveStartSelect();
        } else if (this.isEndSelectAnim) {
            moveEndSelect();
        }
        boolean z2 = (this.isPlay || this.isPause == 2) && !this.isSelect;
        if (!this.isRunning) {
            return;
        }
        try {
            try {
                try {
                    Canvas lockCanvas = this.mHolder.lockCanvas();
                    this.mCanvas = lockCanvas;
                    if (lockCanvas != null) {
                        lockCanvas.drawColor(this.bgColor, PorterDuff.Mode.CLEAR);
                        drawYScale(this.mCanvas);
                        if (this.isSmoothLine) {
                            c = 0;
                            f = 2.0f;
                            i7 = i4;
                            i8 = i5;
                            c2 = 3;
                            i6 = i3;
                            drawLine(this.mCanvas, this.strSmoothDataBuffer, this.mPaths[3], this.mPlayPaths[3], this.mPaints[3], this.mPlayPaints[3], i, this.mPlayX, z2);
                        } else {
                            c = 0;
                            f = 2.0f;
                            c2 = 3;
                            i6 = i3;
                            i7 = i4;
                            i8 = i5;
                        }
                        if (this.isStartPositionLine) {
                            drawLine(this.mCanvas, this.strStartPositionDataBuffer, this.mPaths[2], this.mPlayPaths[2], this.mPaints[2], this.mPlayPaints[2], i, this.mPlayX, z2);
                        }
                        if (this.isVibrateLine) {
                            drawLine(this.mCanvas, this.strVibrateDataBuffer, this.mPaths[1], this.mPlayPaths[1], this.mPaints[1], this.mPlayPaints[1], i, this.mPlayX, z2);
                        }
                        int i28 = i7;
                        drawLine(this.mCanvas, this.strDataBuffer, this.mPaths[c], this.mPlayPaths[c], this.mPaints[c], this.mPlayPaints[c], i, this.mPlayX, z2, this.funDataMap);
                        if (z2) {
                            if (!this.isPlay && !this.isPlayCircle) {
                                if (this.isPause == 2) {
                                    drawSelect(this.mCanvas, this.mPlayX);
                                }
                            }
                            if (this.isSmoothLine) {
                                float f8 = this.mYHeight;
                                this.mCanvas.drawCircle(this.mPlayX, (f8 - ((f8 / 100.0f) * i2)) + ((this.mHeight - f8) / f), this.mStrokeWidth * 1.5f, this.mCirclePaints[c2]);
                            }
                            if (this.isStartPositionLine) {
                                float f9 = this.mYHeight;
                                this.mCanvas.drawCircle(this.mPlayX, (f9 - ((f9 / 100.0f) * i8)) + ((this.mHeight - f9) / f), this.mStrokeWidth * 1.5f, this.mCirclePaints[2]);
                            }
                            if (this.isVibrateLine) {
                                float f10 = this.mYHeight;
                                this.mCanvas.drawCircle(this.mPlayX, (f10 - ((f10 / 100.0f) * i28)) + ((this.mHeight - f10) / f), this.mStrokeWidth * 1.5f, this.mCirclePaints[1]);
                            }
                            float f11 = this.mYHeight;
                            i9 = 0;
                            this.mCanvas.drawCircle(this.mPlayX, (f11 - ((f11 / 100.0f) * i6)) + ((this.mHeight - f11) / f), this.mStrokeWidth * 1.5f, this.mCirclePaints[0]);
                            this.mCanvas.clipRect(i9, i9, this.mWidth, this.mHeight);
                            this.mCanvas.clipRect(0.0f, 0.0f, this.mScaleLeftX - this.mDistance, this.mHeight, Region.Op.DIFFERENCE);
                            if (this.isSelect) {
                                drawCursor(this.mCanvas, i);
                            }
                        }
                        i9 = 0;
                        this.mCanvas.clipRect(i9, i9, this.mWidth, this.mHeight);
                        this.mCanvas.clipRect(0.0f, 0.0f, this.mScaleLeftX - this.mDistance, this.mHeight, Region.Op.DIFFERENCE);
                        if (this.isSelect) {
                        }
                    }
                    canvas = this.mCanvas;
                } catch (Exception e) {
                    e.printStackTrace();
                    canvas = this.mCanvas;
                    if (canvas == null) {
                        return;
                    } else {
                        surfaceHolder = this.mHolder;
                    }
                }
                if (canvas != null) {
                    surfaceHolder = this.mHolder;
                    surfaceHolder.unlockCanvasAndPost(canvas);
                }
            } catch (Exception e2) {
                e2.printStackTrace();
            }
        } finally {
        }
    }

    private void drawAdhesionGroup(Canvas canvas, float f, float f2, float f3) {
        LineDrawView5 lineDrawView5;
        float f4;
        float f5;
        float width = this.mFunPiuIcon.getWidth() / 2.0f;
        float f6 = 0.5f * width;
        float f7 = this.PIU_MAX_DISTANCE;
        if (f3 < f7) {
            float max = Math.max(0.0f, 1.0f - (f3 / f7));
            if (f3 > 1.0f && max > 0.01f) {
                this.mPiuAdhesionPath.reset();
                lineDrawView5 = this;
                f4 = f;
                f5 = f2;
                lineDrawView5.buildAdhesionPath(this.mDragOriX, this.mDragOriY, f6 * max, f4, f5, width * max);
                canvas.drawPath(lineDrawView5.mPiuAdhesionPath, lineDrawView5.mPiuDotPaint);
                canvas.drawCircle(lineDrawView5.mDragOriX, lineDrawView5.mDragOriY, f6, lineDrawView5.mPiuDotPaint);
                canvas.drawBitmap(lineDrawView5.mFunPiuIcon, (Rect) null, new RectF(f4 - width, f5 - width, f4 + width, width + f5), lineDrawView5.mFunPaint);
            }
        }
        lineDrawView5 = this;
        f4 = f;
        f5 = f2;
        canvas.drawCircle(lineDrawView5.mDragOriX, lineDrawView5.mDragOriY, f6, lineDrawView5.mPiuDotPaint);
        canvas.drawBitmap(lineDrawView5.mFunPiuIcon, (Rect) null, new RectF(f4 - width, f5 - width, f4 + width, width + f5), lineDrawView5.mFunPaint);
    }

    private void drawCursor(Canvas canvas, int i) {
        if (canvas != null) {
            int i2 = this.intervalTime;
            float f = this.mDistance;
            int i3 = this.mScaleLeftX;
            float f2 = i3;
            float f3 = f2 + ((((this.mSelectStartTime - i) * 1.0f) / i2) * f);
            float f4 = ((((this.mSelectEndTime - i) * 1.0f) / i2) * f) + i3;
            RectF rectF = new RectF(f3, this.mScaleTopY, f4, this.mScaleBottomY);
            float f5 = this.mCursorBorderRadius;
            canvas.drawRoundRect(rectF, f5, f5, this.mSelectPaint);
            int i4 = this.mThumbWidth;
            int i5 = this.whiteWidth;
            float f6 = ((i4 / 2.0f) + f3) - (i5 / 2.0f);
            float f7 = (f4 - (i4 / 2.0f)) - (i5 / 2.0f);
            int i6 = this.mScaleBottomY;
            float f8 = (i6 - this.whiteHeight) / 2;
            RectF rectF2 = this.mLeftRectF;
            if (rectF2 != null) {
                rectF2.set(f3, this.mScaleTopY, i4 + f3, i6 + (this.mStrokeWidth / 2.0f));
            } else {
                this.mLeftRectF = new RectF(f3, this.mScaleTopY, this.mThumbWidth + f3, this.mScaleBottomY + (this.mStrokeWidth / 2.0f));
            }
            RectF rectF3 = this.mRightRectF;
            if (rectF3 != null) {
                rectF3.set(f4 - this.mThumbWidth, this.mScaleTopY, f4, this.mScaleBottomY + (this.mStrokeWidth / 2.0f));
            } else {
                this.mRightRectF = new RectF(f4 - this.mThumbWidth, this.mScaleTopY, f4, this.mScaleBottomY + (this.mStrokeWidth / 2.0f));
            }
            canvas.drawRect(this.mLeftRectF, this.mCursorPaint);
            canvas.drawRect(this.mRightRectF, this.mCursorPaint);
            canvas.drawRect(f6, f8, f6 + this.whiteWidth, f8 + this.whiteHeight, this.mSelectPaint);
            canvas.drawRect(f7, f8, f7 + this.whiteWidth, f8 + this.whiteHeight, this.mSelectPaint);
            canvas.drawRect(0.0f, this.mScaleTopY, f3, this.mScaleBottomY, this.mDullPaint);
            canvas.drawRect(f4, this.mScaleTopY, this.mWidth, this.mScaleBottomY, this.mDullPaint);
        }
    }

    private void drawLine(Canvas canvas, StringBuffer stringBuffer, Path path, Path path2, Paint paint, Paint paint2, int i, float f, boolean z) {
        drawLine(canvas, stringBuffer, path, path2, paint, paint2, i, f, z, null);
    }

    /* JADX WARN: Removed duplicated region for block: B:19:0x00a3  */
    /* JADX WARN: Removed duplicated region for block: B:32:0x011e  */
    /* JADX WARN: Removed duplicated region for block: B:35:0x0131 A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:38:0x0139  */
    /* JADX WARN: Removed duplicated region for block: B:39:0x013d  */
    /* JADX WARN: Removed duplicated region for block: B:41:0x0142  */
    /* JADX WARN: Removed duplicated region for block: B:56:0x0122  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private void drawLine(Canvas canvas, StringBuffer stringBuffer, Path path, Path path2, Paint paint, Paint paint2, int i, float f, boolean z, Map<Integer, String> map) {
        boolean z2;
        StringBuffer stringBuffer2;
        int i2;
        boolean z3;
        int i3;
        float f2;
        int i4;
        String str;
        if (canvas != null) {
            path.reset();
            path2.reset();
            ArrayList<PiuItem> arrayList = new ArrayList();
            int i5 = this.intervalTime;
            int i6 = i % i5;
            float f3 = ((i6 * 1.0f) / i5) * this.mDistance;
            int i7 = i / i5;
            int i8 = i7 * 3;
            int i9 = i7 * i5;
            if (stringBuffer.length() > i8) {
                z2 = true;
                int ceil = ((((int) Math.ceil(((this.mScreenTime + i6) * 1.0f) / this.intervalTime)) + 1) * 3) + i8;
                stringBuffer2 = ceil <= stringBuffer.length() ? new StringBuffer(stringBuffer.substring(i8, ceil)) : new StringBuffer(stringBuffer.substring(i8));
            } else {
                z2 = true;
                stringBuffer2 = null;
            }
            int i10 = 0;
            while (stringBuffer2 != null && i10 < stringBuffer2.length() / 3) {
                float f4 = ((this.mDistance * i10) - f3) + this.mScaleLeftX;
                int i11 = i10 * 3;
                boolean z4 = z2;
                try {
                    i2 = Integer.valueOf(stringBuffer2.substring(i11, i11 + 2), 16).intValue();
                } catch (Exception unused) {
                    i2 = 0;
                }
                int i12 = this.mScaleLeftX;
                if (f4 >= i12) {
                    i12 = this.mScaleRightX;
                    if (f4 <= i12) {
                        z3 = false;
                        if (z3) {
                            if (i10 < (stringBuffer2.length() / 3) - 1) {
                                int i13 = (i10 + 1) * 3;
                                int intValue = Integer.valueOf(stringBuffer2.substring(i13, i13 + 2), 16).intValue();
                                int abs = Math.abs(intValue - i2);
                                int i14 = this.intervalTime;
                                int i15 = (abs * (i % i14)) / i14;
                                i2 = intValue > i2 ? i2 + i15 : i2 - i15;
                            } else if (i10 == (stringBuffer2.length() / 3) - 1) {
                                int i16 = (i10 - 1) * 3;
                                int intValue2 = Integer.valueOf(stringBuffer2.substring(i16, i16 + 2), 16).intValue();
                                int abs2 = Math.abs(intValue2 - i2);
                                int i17 = this.mScreenTime + i6;
                                i3 = i10;
                                int i18 = this.intervalTime;
                                int i19 = (abs2 * (i17 % i18)) / i18;
                                i2 = intValue2 > i2 ? intValue2 - i19 : intValue2 + i19;
                                float f5 = this.mYHeight;
                                f2 = (f5 - ((f5 / 100.0f) * i2)) + ((this.mHeight - f5) / 2.0f);
                                if (path.isEmpty()) {
                                    path.moveTo(f4, f2);
                                } else {
                                    path.lineTo(f4, f2);
                                }
                                if (f - f4 >= -1.0E-4d && z) {
                                    if (path2.isEmpty()) {
                                        path2.lineTo(f4, f2);
                                    } else {
                                        path2.moveTo(f4, f2);
                                    }
                                }
                                if (map != null && !map.isEmpty() && this.isDrawPiu) {
                                    i4 = (this.intervalTime * i3) + i9;
                                    Integer.toHexString(255).toUpperCase();
                                    String upperCase = Integer.toHexString(Constans.START_TYPE_PIU).toUpperCase();
                                    if (map.containsKey(Integer.valueOf(i4)) && (str = map.get(Integer.valueOf(i4))) != null && str.toUpperCase().startsWith(upperCase)) {
                                        PiuItem piuItem = new PiuItem();
                                        piuItem.actualTime = i4;
                                        piuItem.x = f4;
                                        piuItem.y = f2;
                                        arrayList.add(piuItem);
                                        i10 = i3 + 1;
                                        z2 = z4;
                                    }
                                }
                                i10 = i3 + 1;
                                z2 = z4;
                            }
                        }
                        i3 = i10;
                        float f52 = this.mYHeight;
                        f2 = (f52 - ((f52 / 100.0f) * i2)) + ((this.mHeight - f52) / 2.0f);
                        if (path.isEmpty()) {
                        }
                        if (f - f4 >= -1.0E-4d) {
                            if (path2.isEmpty()) {
                            }
                        }
                        if (map != null) {
                            i4 = (this.intervalTime * i3) + i9;
                            Integer.toHexString(255).toUpperCase();
                            String upperCase2 = Integer.toHexString(Constans.START_TYPE_PIU).toUpperCase();
                            if (map.containsKey(Integer.valueOf(i4))) {
                                PiuItem piuItem2 = new PiuItem();
                                piuItem2.actualTime = i4;
                                piuItem2.x = f4;
                                piuItem2.y = f2;
                                arrayList.add(piuItem2);
                                i10 = i3 + 1;
                                z2 = z4;
                            }
                        }
                        i10 = i3 + 1;
                        z2 = z4;
                    }
                }
                f4 = i12;
                z3 = z4;
                if (z3) {
                }
                i3 = i10;
                float f522 = this.mYHeight;
                f2 = (f522 - ((f522 / 100.0f) * i2)) + ((this.mHeight - f522) / 2.0f);
                if (path.isEmpty()) {
                }
                if (f - f4 >= -1.0E-4d) {
                }
                if (map != null) {
                }
                i10 = i3 + 1;
                z2 = z4;
            }
            canvas.drawPath(path, paint);
            canvas.drawPath(path2, paint2);
            if (map == this.funDataMap) {
                synchronized (this.mPiuLock) {
                    this.mPiuItems = arrayList;
                }
            }
            Bitmap bitmap = this.mFunPiuIcon;
            if (bitmap == null || bitmap.isRecycled() || !this.isDrawPiu) {
                return;
            }
            for (PiuItem piuItem3 : arrayList) {
                if (!isDraggingPiuItem(piuItem3)) {
                    canvas.drawBitmap(this.mFunPiuIcon, piuItem3.x - (this.mFunPiuIcon.getWidth() / 2.0f), piuItem3.y - (this.mFunPiuIcon.getHeight() / 2.0f), this.mFunPaint);
                }
            }
            drawPiuDrag(canvas);
        }
    }

    private void drawPiuDrag(Canvas canvas) {
        PiuItem piuItem;
        if (canvas == null || this.mPiuTouchState == 0 || (piuItem = this.mDragItem) == null) {
            return;
        }
        long currentTimeMillis = System.currentTimeMillis() - this.mPiuAnimStartTime;
        int i = this.intervalTime;
        int i2 = ((int) (currentTimeMillis / i)) + 1;
        int i3 = this.mPiuTouchState;
        if (i3 == 1) {
            drawAdhesionGroup(canvas, this.mDragX, this.mDragY, getPiuDragDistance());
            return;
        }
        if (i3 != 2) {
            if (i3 == 3) {
                if (i2 < 350 / i) {
                    drawPiuExplosion(canvas, i2);
                    return;
                }
                removePiuItem(piuItem);
                invalidateThumb();
                this.mPiuTouchState = 0;
                this.mDragItem = null;
                return;
            }
            return;
        }
        if (i2 >= 200 / i) {
            drawAdhesionGroup(canvas, this.mDragOriX, this.mDragOriY, 0.0f);
            this.mPiuTouchState = 0;
            this.mDragItem = null;
            return;
        }
        float f = 1.0f - (((i2 * i) * 1.0f) / 200.0f);
        float f2 = 1.0f - (f * f);
        float f3 = this.mSettleFromX;
        float f4 = f3 + ((this.mDragOriX - f3) * f2);
        float f5 = this.mSettleFromY;
        drawAdhesionGroup(canvas, f4, f5 + ((this.mDragOriY - f5) * f2), (float) Math.hypot(f4 - r2, r3 - r4));
    }

    private void drawPiuExplosion(Canvas canvas, int i) {
        float f = ((this.intervalTime * i) * 1.0f) / 350.0f;
        float f2 = f > 1.0f ? 1.0f : f;
        float width = this.mFunPiuIcon.getWidth();
        float f3 = 1.0f - f2;
        this.mPiuPiecePaint.setAlpha((int) (180.0f * f3 * f3));
        canvas.drawCircle(this.mDragX, this.mDragY, ((0.45f * f2) + 0.22f) * width, this.mPiuPiecePaint);
        float max = Math.max(0.0f, f2 - 0.13f);
        int i2 = 0;
        while (i2 < 18) {
            int i3 = i2 + 1;
            float sin = ((float) Math.sin(i3 * 12.9898f)) * 43758.547f;
            float floor = sin - ((float) Math.floor(sin));
            float f4 = ((1.2f * floor) + 1.5f) * width;
            float f5 = f4 * f2;
            float f6 = f4 * max;
            double d = (float) (((i2 * 6.283185307179586d) / 18.0d) + ((floor - 0.5f) * 0.28f));
            float f7 = f2;
            float cos = this.mDragX + (((float) Math.cos(d)) * f5);
            float sin2 = this.mDragY + (((float) Math.sin(d)) * f5);
            float f8 = 1.1f * width;
            float f9 = sin2 + (f8 * f7 * f7);
            float cos2 = this.mDragX + (((float) Math.cos(d)) * f6);
            float sin3 = this.mDragY + (((float) Math.sin(d)) * f6) + (f8 * max * max);
            int i4 = (int) (255.0f * f3 * ((floor * 0.35f) + 0.65f));
            this.mPiuSparkPaint.setAlpha(i4);
            this.mPiuSparkPaint.setStrokeWidth(Math.max(1.0f, (0.1f - (0.04f * f7)) * width));
            canvas.drawLine(cos2, sin3, cos, f9, this.mPiuSparkPaint);
            this.mPiuPiecePaint.setAlpha(i4);
            canvas.drawCircle(cos, f9, Math.max(1.0f, (0.1f - (f7 * 0.055f)) * width), this.mPiuPiecePaint);
            i2 = i3;
            f2 = f7;
        }
        this.mPiuPiecePaint.setAlpha(255);
        this.mPiuSparkPaint.setAlpha(255);
    }

    private void drawScale(Canvas canvas, int i) {
        if (canvas != null) {
            float f = this.mScaleLeftX - (this.mStrokeWidth / 2.0f);
            int i2 = this.mScaleBottomY;
            canvas.drawLine(f, i2, this.mScaleRightX, i2, this.mScalePaint);
            int i3 = this.intervalTime;
            float f2 = this.mDistance;
            float f3 = (((i % i3) * 1.0f) / i3) * f2;
            float f4 = ((i / i3) * f2) % (this.mScaleItemWidth * this.mScaleShortNum);
            for (int i4 = 0; i4 <= this.mScaleNum; i4++) {
                float f5 = ((i4 * this.mScaleItemWidth) - f4) - f3;
                int i5 = this.mScaleLeftX;
                float f6 = f5 + i5;
                if (f6 >= i5 && f6 <= this.mScaleRightX) {
                    int i6 = i4 % this.mScaleShortNum;
                    int i7 = this.mScaleBottomY;
                    if (i6 == 0) {
                        canvas.drawLine(f6, i7, f6, i7 - this.mXScaleLongLine, this.mScalePaint);
                    } else {
                        canvas.drawLine(f6, i7, f6, i7 - this.mXScaleShortLine, this.mScalePaint);
                    }
                }
            }
        }
    }

    private void drawSelect(Canvas canvas, float f) {
        if (canvas != null) {
            int i = this.mSelectLineWidth;
            canvas.drawRect(f - (i / 2.0f), this.mScaleTopY, f + (i / 2.0f), this.mScaleBottomY, this.mCursorPaint);
            int i2 = this.mSelectLineWidth;
            int i3 = this.mScaleTopY;
            canvas.drawLine(f - (i2 / 2.0f), i3, f + (i2 / 2.0f), i3, this.mSelectPaint);
            int i4 = this.mSelectLineWidth;
            int i5 = this.mScaleBottomY;
            canvas.drawLine(f - (i4 / 2.0f), i5, f + (i4 / 2.0f), i5, this.mSelectPaint);
            canvas.drawLine(f, this.mScaleTopY, f, this.mScaleBottomY, this.mSelectPaint);
        }
    }

    private void drawYScale(Canvas canvas) {
        Canvas canvas2;
        if (canvas == null || !this.isDrawYScale) {
            return;
        }
        int i = 0;
        int i2 = 0;
        while (i <= this.mYItemNum) {
            float f = (this.mYItemHeight * i) + ((this.mHeight - this.mYHeight) / 2.0f);
            if (i % 2 == 0) {
                canvas2 = canvas;
                canvas2.drawLine(this.mScaleLeftX, f, this.mScaleRightX, f, this.mEffectPaint);
                canvas2.drawText(this.YData[i2], this.mLineTextIntersticeWidth, f + (this.mTextHeight / 2.0f), this.mTextPaint);
                i2++;
            } else {
                canvas2 = canvas;
            }
            i++;
            canvas = canvas2;
        }
    }

    private void drawYScale2(Canvas canvas) {
        if (canvas != null) {
            float f = this.mScaleLeftX;
            int i = this.mHeight;
            float f2 = this.mYHeight;
            canvas.drawLine(f, (i - f2) / 2.0f, f, ((i - f2) / 2.0f) + f2, this.mScalePaint);
            int i2 = 0;
            for (int i3 = 0; i3 <= this.mYItemNum; i3++) {
                float f3 = (this.mYItemHeight * i3) + ((this.mHeight - this.mYHeight) / 2.0f);
                int i4 = i3 % 2;
                int i5 = this.mScaleLeftX;
                if (i4 == 0) {
                    float f4 = this.mStrokeWidth;
                    canvas.drawLine(i5 - (f4 / 2.0f), f3, (i5 - (f4 / 2.0f)) + this.mYScaleLongLine, f3, this.mScalePaint);
                    canvas.drawText(this.YData[i2], this.mScaleLeftX + this.mYScaleLongLine + this.mLineTextIntersticeWidth, f3 + (this.mTextHeight / 2.0f), this.mTextPaint);
                    i2++;
                } else {
                    float f5 = this.mStrokeWidth;
                    canvas.drawLine(i5 - (f5 / 2.0f), f3, (i5 - (f5 / 2.0f)) + this.mYScaleShortLine, f3, this.mScalePaint);
                }
            }
        }
    }

    private float getPiuDragDistance() {
        float f = this.mDragX - this.mDragOriX;
        float f2 = this.mDragY - this.mDragOriY;
        return (float) Math.sqrt((f * f) + (f2 * f2));
    }

    private boolean grabPiuItem(float f, float f2) {
        PiuItem piuItem;
        if (!this.isPiuDragRemoveEnabled || !this.isDrawPiu || this.mScaleItemWidth <= 0.0f) {
            return false;
        }
        synchronized (this.mPiuLock) {
            piuItem = null;
            float f3 = Float.MAX_VALUE;
            for (PiuItem piuItem2 : this.mPiuItems) {
                float f4 = f - piuItem2.x;
                float f5 = f2 - piuItem2.y;
                float sqrt = (float) Math.sqrt((f4 * f4) + (f5 * f5));
                if (sqrt <= this.PIU_HIT_RADIUS && sqrt < f3) {
                    piuItem = piuItem2;
                    f3 = sqrt;
                }
            }
        }
        if (piuItem == null) {
            return false;
        }
        this.mDragItem = piuItem;
        this.mDragOriX = piuItem.x;
        this.mDragOriY = piuItem.y;
        this.mDragTouchOffsetX = piuItem.x - f;
        this.mDragTouchOffsetY = piuItem.y - f2;
        this.mDragX = piuItem.x;
        this.mDragY = piuItem.y;
        this.mPiuTouchState = 1;
        return true;
    }

    private boolean inInTarget(RectF rectF, float f, float f2) {
        if (rectF != null) {
            return rectF.contains(f, f2);
        }
        return false;
    }

    private void init() {
        SurfaceHolder holder = getHolder();
        this.mHolder = holder;
        holder.addCallback(this);
        setFocusable(true);
        setFocusableInTouchMode(true);
        setKeepScreenOn(true);
        setZOrderOnTop(true);
        this.mHolder.setFormat(-2);
        Paint paint = new Paint();
        this.mTextPaint = paint;
        paint.setAntiAlias(true);
        this.mTextPaint.setStyle(Paint.Style.FILL);
        this.mTextPaint.setTextSize(this.mTextSize);
        this.mTextPaint.setColor(this.textColor);
        Paint paint2 = new Paint();
        this.mScalePaint = paint2;
        paint2.setAntiAlias(true);
        this.mScalePaint.setStyle(Paint.Style.STROKE);
        this.mScalePaint.setStrokeWidth(this.mStrokeWidth);
        this.mScalePaint.setColor(-7829368);
        Paint paint3 = new Paint();
        this.mCursorPaint = paint3;
        paint3.setAntiAlias(true);
        this.mCursorPaint.setStyle(Paint.Style.FILL);
        this.mCursorPaint.setColor(this.cursorColor);
        Paint paint4 = new Paint();
        this.mSelectPaint = paint4;
        paint4.setAntiAlias(true);
        this.mSelectPaint.setStyle(Paint.Style.STROKE);
        this.mSelectPaint.setColor(this.selectColor);
        this.mSelectPaint.setStrokeWidth(this.mStrokeWidth);
        this.mSelectPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mSelectPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint5 = new Paint();
        this.mDullPaint = paint5;
        paint5.setAntiAlias(true);
        this.mDullPaint.setStyle(Paint.Style.FILL);
        this.mDullPaint.setColor(this.dullColor);
        Paint paint6 = new Paint();
        this.mEffectPaint = paint6;
        paint6.setAntiAlias(true);
        this.mEffectPaint.setStyle(Paint.Style.STROKE);
        this.mEffectPaint.setStrokeWidth(this.mStrokeWidth);
        this.mEffectPaint.setColor(this.effectColor);
        int i = 0;
        this.mEffectPaint.setPathEffect(new DashPathEffect(new float[]{MeasureUtil.dip2px(getContext(), 6.0f), MeasureUtil.dip2px(getContext(), 4.0f)}, 0.0f));
        this.mPaints = new Paint[4];
        this.mPaths = new Path[4];
        this.mPlayPaints = new Paint[4];
        this.mCirclePaints = new Paint[4];
        this.mPlayPaths = new Path[4];
        while (true) {
            Paint[] paintArr = this.mPaints;
            if (i >= paintArr.length) {
                Bitmap decodeResource = BitmapFactory.decodeResource(getResources(), R.mipmap.ic_iplay_piu);
                this.mFunPiuIcon = decodeResource;
                int i2 = (int) (this.mStrokeWidth * 8.0f);
                this.PIU_HIT_RADIUS = i2 * 0.5f;
                this.mFunPiuIcon = Bitmap.createScaledBitmap(decodeResource, i2, i2, true);
                Paint paint7 = new Paint();
                this.mFunPaint = paint7;
                paint7.setAntiAlias(true);
                this.mFunPaint.setFilterBitmap(true);
                this.mFunPaint.setAlpha(255);
                this.mPiuAdhesionPath = new Path();
                Paint paint8 = new Paint();
                this.mPiuDotPaint = paint8;
                paint8.setAntiAlias(true);
                this.mPiuDotPaint.setStyle(Paint.Style.FILL);
                this.mPiuDotPaint.setColor(this.PIU_COLOR);
                Paint paint9 = new Paint();
                this.mPiuPiecePaint = paint9;
                paint9.setAntiAlias(true);
                this.mPiuPiecePaint.setStyle(Paint.Style.FILL);
                this.mPiuPiecePaint.setColor(this.PIU_COLOR);
                Paint paint10 = new Paint();
                this.mPiuSparkPaint = paint10;
                paint10.setAntiAlias(true);
                this.mPiuSparkPaint.setStyle(Paint.Style.STROKE);
                this.mPiuSparkPaint.setStrokeCap(Paint.Cap.ROUND);
                this.mPiuSparkPaint.setColor(this.PIU_COLOR);
                this.mTextHeight = MeasureUtil.calcTextHeight(this.mTextPaint, "0");
                setDrawYScale(true);
                this.funDataMap = new LinkedHashMap();
                this.funVibrateDataMap = new LinkedHashMap();
                return;
            }
            paintArr[i] = new Paint();
            this.mPaints[i].setAntiAlias(true);
            this.mPaints[i].setStyle(Paint.Style.STROKE);
            this.mPaints[i].setStrokeWidth(this.mStrokeWidth);
            this.mPaints[i].setStrokeJoin(Paint.Join.ROUND);
            this.mPaints[i].setStrokeCap(Paint.Cap.ROUND);
            this.mPaths[i] = new Path();
            this.mPlayPaints[i] = new Paint();
            this.mPlayPaints[i].setAntiAlias(true);
            this.mPlayPaints[i].setStyle(Paint.Style.STROKE);
            this.mPlayPaints[i].setStrokeWidth(this.mStrokeWidth);
            if (i == 0) {
                this.mPlayPaints[i].setColor(getResources().getColor(R.color.play_mode_line_circle));
            } else if (i == 1) {
                this.mPlayPaints[i].setColor(getResources().getColor(R.color.play_mode_vibrate_line_circle));
            } else if (i == 2) {
                this.mPlayPaints[i].setColor(getResources().getColor(R.color.play_mode_start_position_line_circle));
            } else if (i == 3) {
                this.mPlayPaints[i].setColor(getResources().getColor(R.color.play_mode_smooth_line_circle));
            }
            this.mPlayPaints[i].setStrokeJoin(Paint.Join.ROUND);
            this.mPlayPaints[i].setStrokeCap(Paint.Cap.ROUND);
            this.mCirclePaints[i] = new Paint();
            this.mCirclePaints[i].setAntiAlias(true);
            this.mCirclePaints[i].setStyle(Paint.Style.FILL);
            if (i == 0) {
                this.mCirclePaints[i].setColor(getResources().getColor(R.color.play_mode_line_circle));
            } else if (i == 1) {
                this.mCirclePaints[i].setColor(getResources().getColor(R.color.play_mode_vibrate_line_circle));
            } else if (i == 2) {
                this.mCirclePaints[i].setColor(getResources().getColor(R.color.play_mode_start_position_line_circle));
            } else if (i == 3) {
                this.mCirclePaints[i].setColor(getResources().getColor(R.color.play_mode_smooth_line_circle));
            }
            this.mPlayPaths[i] = new Path();
            i++;
        }
    }

    private void initAttr(AttributeSet attributeSet) {
        if (attributeSet != null) {
            TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, R.styleable.LineDrawView5);
            obtainStyledAttributes.getColor(R.styleable.LineDrawView5_startColor, -1);
            obtainStyledAttributes.getColor(R.styleable.LineDrawView5_endColor, -1);
            obtainStyledAttributes.getColor(R.styleable.LineDrawView5_playColor, -1);
            obtainStyledAttributes.getColor(R.styleable.LineDrawView5_startColor2, -1);
            obtainStyledAttributes.getColor(R.styleable.LineDrawView5_endColor2, -1);
            obtainStyledAttributes.getColor(R.styleable.LineDrawView5_playColor2, -1);
            this.textColor = obtainStyledAttributes.getColor(R.styleable.LineDrawView5_textColor, Color.parseColor("#7FFFFFFF"));
            this.effectColor = obtainStyledAttributes.getColor(R.styleable.LineDrawView5_effectColor, Color.parseColor("#18FFFFFF"));
            this.dullColor = obtainStyledAttributes.getColor(R.styleable.LineDrawView5_dullColor, Color.parseColor("#99000000"));
            this.selectColor = obtainStyledAttributes.getColor(R.styleable.LineDrawView5_selectColor, Color.parseColor("#FFFFFF"));
            this.cursorColor = obtainStyledAttributes.getColor(R.styleable.LineDrawView5_cursorColor, Color.parseColor("#33FFFFFF"));
            obtainStyledAttributes.recycle();
        }
    }

    private void initFunData() {
        try {
            String upperCase = Integer.toHexString(255).toUpperCase();
            String upperCase2 = Integer.toHexString(Constans.START_TYPE_PIU).toUpperCase();
            if (this.strDataBuffer.toString().contains(upperCase) || this.strDataBuffer.toString().contains(upperCase2)) {
                String[] split = this.strDataBuffer.toString().split(this.mSeparator);
                this.strDataBuffer = new StringBuffer();
                int i = 0;
                for (int i2 = 0; i2 < split.length; i2++) {
                    if (!split[i2].toUpperCase().startsWith(upperCase) && !split[i2].toUpperCase().startsWith(upperCase2)) {
                        this.strDataBuffer.append(split[i2]);
                        this.strDataBuffer.append(this.mSeparator);
                        i++;
                    } else if (i > 0) {
                        this.funDataMap.put(Integer.valueOf((i - 1) * this.intervalTime), split[i2]);
                    }
                }
            }
            if (this.strVibrateDataBuffer.toString().contains(upperCase)) {
                String[] split2 = this.strVibrateDataBuffer.toString().split(this.mSeparator);
                this.strVibrateDataBuffer = new StringBuffer();
                int i3 = 0;
                for (int i4 = 0; i4 < split2.length; i4++) {
                    if (!split2[i4].toUpperCase().startsWith(upperCase)) {
                        this.strVibrateDataBuffer.append(split2[i4]);
                        this.strVibrateDataBuffer.append(this.mSeparator);
                        i3++;
                    } else if (i3 > 0) {
                        this.funVibrateDataMap.put(Integer.valueOf((i3 - 1) * this.intervalTime), split2[i4]);
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void invalidateThumb() {
        int i = this.mPlayTime;
        int i2 = this.mTime;
        if (i > i2) {
            this.mPlayTime = i2;
        }
        this.mHandler.sendEmptyMessage(1);
        sendStatus();
    }

    private boolean isDraggingPiuItem(PiuItem piuItem) {
        return (piuItem == null || this.mPiuTouchState == 0 || this.mDragItem == null || piuItem.actualTime != this.mDragItem.actualTime) ? false : true;
    }

    /* JADX WARN: Removed duplicated region for block: B:16:0x0047 A[Catch: all -> 0x0177, TryCatch #0 {, blocks: (B:4:0x0005, B:9:0x000b, B:11:0x0015, B:16:0x0047, B:17:0x0081, B:20:0x003c, B:23:0x0091, B:25:0x0095, B:27:0x0099, B:29:0x009d, B:31:0x00a1, B:35:0x00a9, B:36:0x00bb, B:38:0x00bf, B:40:0x00db, B:42:0x00e5, B:43:0x00e8, B:45:0x00ec, B:46:0x00ef, B:48:0x00f3, B:49:0x010b, B:50:0x0173, B:51:0x00f8, B:53:0x0100, B:54:0x0106, B:55:0x013e, B:57:0x0142, B:59:0x014e, B:60:0x0151, B:62:0x0155, B:63:0x0158, B:65:0x0160, B:66:0x0165, B:68:0x016a, B:69:0x016f), top: B:3:0x0005 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private synchronized void move(MotionEvent motionEvent) {
        if (this.mPiuTouchState != 0) {
            return;
        }
        float f = 1.0f;
        if (motionEvent.getPointerCount() > 1) {
            float abs = Math.abs(motionEvent.getX(1) - motionEvent.getX(0));
            float abs2 = Math.abs(this.downTwoX - this.downOneX);
            this.distance = abs2;
            float f2 = ((abs - abs2) / abs2) + 1.0f;
            float f3 = this.mScaleItemWidth;
            float f4 = f2 * f3;
            float f5 = this.mMaxScaleItemWidth;
            if (f4 <= f5) {
                f5 = this.mMinScaleItemWidth;
                if (f4 < f5) {
                }
                if (f4 != f3) {
                    LogUtil.e("width：" + f4);
                    this.mScaleItemWidth = f4;
                    int i = this.mScaleWidth;
                    int i2 = this.mScaleShortNum;
                    this.mScaleNum = (int) ((i / f4) + (i2 * 2));
                    int i3 = this.intervalTime;
                    float f6 = (f4 * i2) / (1000.0f / i3);
                    this.mDistance = f6;
                    this.mScreenTime = (int) ((i / f6) * i3);
                    this.mThumbTime = (int) ((this.mThumbWidth / f6) * i3);
                }
                this.downOneX = motionEvent.getX(0);
                this.downTwoX = motionEvent.getX(1);
                this.distance = abs;
            }
            f4 = f5;
            if (f4 != f3) {
            }
            this.downOneX = motionEvent.getX(0);
            this.downTwoX = motionEvent.getX(1);
            this.distance = abs;
        } else if (this.needDelay) {
            if (this.inRectTarget > 0) {
                float x = motionEvent.getX(0);
                float f7 = this.downOneX - x;
                float f8 = this.mThumbOffX + f7;
                float f9 = this.mDistance;
                int i4 = (int) (f8 / f9);
                this.mThumbOffX = (f8 + f7) % f9;
                int i5 = this.inRectTarget;
                if ((i5 & 1) == 1) {
                    int i6 = this.mSelectStartTime - (this.intervalTime * i4);
                    this.mSelectStartTime = i6;
                    if (i6 < 0) {
                        this.mSelectStartTime = 0;
                        i6 = 0;
                    }
                    int i7 = this.mSelectEndTime;
                    if (i6 > i7) {
                        this.mSelectStartTime = i7;
                        i6 = i7;
                    }
                    int i8 = this.mShowTime;
                    if (i6 < i8) {
                        this.isStartSelectAnimMoveTime = 1.0f;
                        this.isStartSelectAnim = true;
                    } else if (i6 + this.mThumbTime > i8 + this.mScreenTime) {
                        this.isStartSelectAnimMoveTime = -1.0f;
                        this.isStartSelectAnim = true;
                        f = -1.0f;
                    } else {
                        this.isStartSelectAnimMoveTime = 0.0f;
                        this.isStartSelectAnim = false;
                        f = 0.0f;
                    }
                    LogUtil.e("moveDis：" + f + ", moveTime:" + i4 + ", isStartSelectAnimMoveTime:" + this.isStartSelectAnimMoveTime + ", isStartSelectAnim:" + this.isStartSelectAnim);
                } else if ((i5 & 2) == 2) {
                    int i9 = this.mSelectEndTime - (i4 * this.intervalTime);
                    this.mSelectEndTime = i9;
                    int i10 = this.mSelectStartTime;
                    if (i9 < i10) {
                        this.mSelectEndTime = i10;
                        i9 = i10;
                    }
                    int i11 = this.mTime;
                    if (i9 > i11) {
                        this.mSelectEndTime = i11;
                        i9 = i11;
                    }
                    int i12 = i9 - this.mThumbTime;
                    int i13 = this.mShowTime;
                    if (i12 < i13) {
                        this.isEndSelectAnimMoveTime = 1.0f;
                        this.isEndSelectAnim = true;
                    } else if (i9 > i13 + this.mScreenTime) {
                        this.isEndSelectAnimMoveTime = -1.0f;
                        this.isEndSelectAnim = true;
                    } else {
                        this.isEndSelectAnimMoveTime = 0.0f;
                        this.isEndSelectAnim = false;
                    }
                }
                this.downOneX = x;
            }
        } else if (!this.isRecord && !this.isPlay) {
            if (this.isSelect && this.mTime < this.mScreenTime) {
                return;
            }
            moveX(this.downOneX - motionEvent.getX(0));
            this.downOneX = motionEvent.getX(0);
        }
    }

    private void moveEndSelect() {
        float f = this.mSelectEndTime;
        int i = this.intervalTime;
        float f2 = this.isEndSelectAnimMoveTime;
        int i2 = (int) (f - ((i * 10) * f2));
        this.mSelectEndTime = i2;
        int i3 = (int) (this.mPlayTime - ((i * 10) * f2));
        this.mPlayTime = i3;
        int i4 = this.mTime;
        int i5 = this.mScreenTime;
        if (i3 > i4 - (i5 / 2)) {
            this.mPlayTime = i4 - (i5 / 2);
        }
        if (i2 > i4) {
            this.mSelectEndTime = i4;
            this.isEndSelectAnim = false;
            i2 = i4;
        }
        int i6 = this.mSelectStartTime;
        if (i2 < i6) {
            this.mSelectEndTime = i6;
            this.isEndSelectAnim = false;
        }
    }

    private void movePiuItem(float f, float f2) {
        if (this.mDragItem == null) {
            return;
        }
        Bitmap bitmap = this.mFunPiuIcon;
        float width = (bitmap == null || bitmap.isRecycled()) ? 0.0f : this.mFunPiuIcon.getWidth() / 2.0f;
        float f3 = f + this.mDragTouchOffsetX;
        float f4 = f2 + this.mDragTouchOffsetY;
        int i = this.mScaleLeftX;
        if (f3 < i + width) {
            f3 = i + width;
        } else {
            int i2 = this.mScaleRightX;
            if (f3 > i2 - width) {
                f3 = i2 - width;
            }
        }
        int i3 = this.mScaleTopY;
        if (f4 < i3 + width) {
            f4 = i3 + width;
        } else {
            int i4 = this.mScaleBottomY;
            if (f4 > i4 - width) {
                f4 = i4 - width;
            }
        }
        this.mDragX = f3;
        this.mDragY = f4;
    }

    private void moveStartSelect() {
        float f = this.mSelectStartTime;
        int i = this.intervalTime;
        float f2 = this.isStartSelectAnimMoveTime;
        int i2 = (int) (f - ((i * 10) * f2));
        this.mSelectStartTime = i2;
        int i3 = (int) (this.mPlayTime - ((i * 10) * f2));
        this.mPlayTime = i3;
        int i4 = this.mScreenTime;
        if (i3 < i4 / 2) {
            this.mPlayTime = i4 / 2;
        }
        if (i2 < 0) {
            this.mSelectStartTime = 0;
            this.isStartSelectAnim = false;
            i2 = 0;
        }
        int i5 = this.mSelectEndTime;
        if (i2 > i5) {
            this.mSelectStartTime = i5;
            this.isStartSelectAnim = false;
        }
    }

    private void moveX(float f) {
        float f2 = this.mOffX + f;
        float f3 = this.mDistance;
        int i = (int) (f2 / f3);
        this.mOffX = (f2 + f) % f3;
        if (this.isPause == 0) {
            this.mPlayTime = this.mShowTime;
            this.isPause = 2;
            sendStatus();
        }
        if (this.isPause == 1) {
            this.mPlayTime = this.mTime;
            this.isPause = 2;
            sendStatus();
        }
        int i2 = this.mPlayTime + (i * this.intervalTime);
        this.mPlayTime = i2;
        if (this.isSelect) {
            int i3 = this.mScreenTime;
            if (i2 < i3 / 2) {
                i2 = i3 / 2;
                this.mPlayTime = i2;
            } else {
                int i4 = this.mTime;
                if (i2 > i4 - (i3 / 2)) {
                    i2 = i4 - (i3 / 2);
                    this.mPlayTime = i2;
                }
            }
        }
        if (i2 < 0) {
            i2 = 0;
            this.mPlayTime = 0;
        }
        int i5 = this.mTime;
        if (i2 > i5) {
            this.mPlayTime = i5;
        }
    }

    private void releasePiuItem() {
        if (this.mDragItem == null) {
            this.mPiuTouchState = 0;
            return;
        }
        float piuDragDistance = getPiuDragDistance();
        this.mPiuAnimStartTime = System.currentTimeMillis();
        if (piuDragDistance >= this.PIU_MAX_DISTANCE) {
            this.mPiuTouchState = 3;
            LogUtil.e("PIU拖拽删除 distance：" + piuDragDistance);
        } else {
            this.mSettleFromX = this.mDragX;
            this.mSettleFromY = this.mDragY;
            this.mPiuTouchState = 2;
            LogUtil.e("PIU拖拽回弹 distance：" + piuDragDistance);
        }
    }

    /* JADX WARN: Code restructure failed: missing block: B:15:0x0047, code lost:
    
        r5 = r4 + r10;
        r0.put(java.lang.Integer.valueOf(r5), r3.getValue());
        com.hismith.hismithapp.utils.LogUtil.e("paste PIU功能点时间调整 key：" + r4 + " -> " + r5);
     */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private void remapFunDataMapAfterPaste(int i, int i2, int i3) {
        int i4 = this.intervalTime;
        int i5 = i2 * i4;
        int i6 = (i3 * i4) - i5;
        int i7 = i5 + i;
        TreeMap treeMap = new TreeMap();
        Iterator<Map.Entry<Integer, String>> it = this.funDataMap.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<Integer, String> next = it.next();
            int intValue = next.getKey().intValue();
            if (i2 <= 0 || intValue < i || intValue >= i7) {
                treeMap.put(Integer.valueOf(intValue), next.getValue());
            } else {
                LogUtil.e("paste 删除PIU功能点 key：" + intValue);
            }
        }
        for (Map.Entry<Integer, String> entry : this.mCopyFunDataMap.entrySet()) {
            treeMap.put(Integer.valueOf(entry.getKey().intValue() + i), entry.getValue());
        }
        this.funDataMap.clear();
        this.funDataMap.putAll(treeMap);
    }

    private void removeFunDataInRange(int i, int i2) {
        Map<Integer, String> map = this.funDataMap;
        if (map == null || map.isEmpty()) {
            return;
        }
        int i3 = this.intervalTime;
        int i4 = ((i2 - i) / i3) * i3;
        LinkedHashMap linkedHashMap = new LinkedHashMap();
        for (Map.Entry<Integer, String> entry : this.funDataMap.entrySet()) {
            int intValue = entry.getKey().intValue();
            String value = entry.getValue();
            if (intValue >= i && intValue <= i2) {
                LogUtil.e("删除PIU功能点 key：" + intValue);
            } else if (intValue > i2) {
                int i5 = intValue - i4;
                linkedHashMap.put(Integer.valueOf(i5), value);
                LogUtil.e("PIU功能点时间调整 key：" + intValue + " -> " + i5);
            } else {
                linkedHashMap.put(Integer.valueOf(intValue), value);
            }
        }
        this.funDataMap.clear();
        this.funDataMap.putAll(linkedHashMap);
        for (Map.Entry<Integer, String> entry2 : this.funDataMap.entrySet()) {
            LogUtil.e("剩余PIU功能点 key:" + entry2.getKey().intValue() + ", obj:" + entry2.getValue());
        }
    }

    private void removePiuItem(PiuItem piuItem) {
        if (piuItem == null) {
            return;
        }
        this.funDataMap.remove(Integer.valueOf(piuItem.actualTime));
        LogUtil.e("删除PIU功能点 actualTime：" + piuItem.actualTime + ", mTime：" + this.mTime);
    }

    private void replenishData(StringBuffer stringBuffer) {
        if (TextUtils.isEmpty(stringBuffer) || stringBuffer.length() < this.strDataBuffer.length()) {
            int length = this.strDataBuffer.length() - stringBuffer.length();
            for (int i = 0; i < length / 3; i++) {
                stringBuffer.append("00");
                stringBuffer.append(this.mSeparator);
            }
        }
    }

    private void sendComplete() {
        if (this.mHandler.hasMessages(4)) {
            this.mHandler.removeMessages(4);
        }
        this.mHandler.sendEmptyMessage(4);
    }

    private void sendFunData(String str) {
        Message message = new Message();
        message.what = 8;
        message.obj = str;
        this.mHandler.sendMessage(message);
    }

    private void sendPause() {
        LogUtil.e("sendPause");
        Message message = new Message();
        message.what = 7;
        this.mHandler.sendMessage(message);
        this.lastSpeed = 0;
        this.lastVibrate = 0;
    }

    private void sendPlayFunData() {
        Map<Integer, String> map = this.funDataMap;
        if (map != null && map.containsKey(Integer.valueOf(this.mPlayTime)) && this.isDrawPiu) {
            String str = this.funDataMap.get(Integer.valueOf(this.mPlayTime));
            Message message = new Message();
            message.what = 8;
            message.obj = str;
            this.mHandler.sendMessageDelayed(message, 50L);
        }
    }

    private void sendPlayTime() {
        Message message = new Message();
        message.what = 2;
        message.obj = Integer.valueOf(this.mPlayTime);
        this.mHandler.sendMessage(message);
    }

    private void sendPositionRange(int i, int i2) {
        if (this.isStartPositionLine) {
            LogUtil.e("sendStartPosition:" + i);
            Message message = new Message();
            message.what = 9;
            message.arg1 = i;
            message.arg2 = i2;
            this.mHandler.sendMessageDelayed(message, 20L);
            this.lastStartPosition = i;
            this.lastVibrate = i2;
        }
    }

    private void sendSmooth(int i) {
        if (this.isSmoothLine) {
            LogUtil.e("sendSmooth:" + i);
            Message message = new Message();
            message.what = 16;
            message.obj = Integer.valueOf(i);
            this.mHandler.sendMessageDelayed(message, 20L);
            this.lastSmooth = i;
        }
    }

    private void sendSpeed(int i) {
        LogUtil.d("sendSpeed:" + i);
        Message message = new Message();
        message.what = 5;
        message.obj = Integer.valueOf(i);
        this.mHandler.sendMessage(message);
        this.lastSpeed = i;
    }

    private void sendStatus() {
        if (this.mHandler.hasMessages(3)) {
            this.mHandler.removeMessages(3);
        }
        this.mHandler.sendEmptyMessage(3);
    }

    private void sendVibrate(int i) {
        if (this.isVibrateLine) {
            LogUtil.e("sendVibrate:" + i);
            Message message = new Message();
            message.what = 6;
            message.obj = Integer.valueOf(i);
            this.mHandler.sendMessageDelayed(message, 20L);
            this.lastVibrate = i;
        }
    }

    private void setModeData(int i) {
        String str = Integer.toHexString(i) + this.mSeparator;
        if (str.length() < 3) {
            str = "0" + str;
        }
        this.strDataBuffer.append(str);
    }

    private void setSmoothModeData(int i) {
        String str = Integer.toHexString(i) + this.mSeparator;
        if (str.length() < 3) {
            str = "0" + str;
        }
        this.strSmoothDataBuffer.append(str);
    }

    private void setStartPositionModeData(int i) {
        String str = Integer.toHexString(i) + this.mSeparator;
        if (str.length() < 3) {
            str = "0" + str;
        }
        this.strStartPositionDataBuffer.append(str);
    }

    private void setVibrateModeData(int i) {
        String str = Integer.toHexString(i) + this.mSeparator;
        if (str.length() < 3) {
            str = "0" + str;
        }
        this.strVibrateDataBuffer.append(str);
    }

    public void addData(String str, String str2, String str3, String str4) {
        if (TextUtils.isEmpty(str3) || TextUtils.isEmpty(str4)) {
            this.mType = 1;
        } else {
            this.mType = 2;
        }
        ValueAnimator valueAnimator = this.mPlayAnimator;
        if (valueAnimator != null && valueAnimator.isRunning()) {
            this.mPlayAnimator.end();
        }
        int i = this.mPlayTime;
        int i2 = this.mTime;
        if (i > i2) {
            this.mPlayTime = i2;
            i = i2;
        }
        int i3 = (i / this.intervalTime) + 1;
        final int length = (str.length() / 3) * this.intervalTime;
        int i4 = i3 * 3;
        this.strDataBuffer.insert(i4, str);
        this.strVibrateDataBuffer.insert(i4, str2);
        this.strStartPositionDataBuffer.insert(i4, str3);
        this.strSmoothDataBuffer.insert(i4, str4);
        initFunData();
        replenishData(this.strVibrateDataBuffer);
        replenishData(this.strStartPositionDataBuffer);
        replenishData(this.strSmoothDataBuffer);
        this.mTime = ((this.strDataBuffer.length() / 3) - 1) * this.intervalTime;
        this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.surface_view.LineDrawView5.1
            @Override // java.lang.Runnable
            public void run() {
                final int i5 = LineDrawView5.this.mPlayTime;
                LineDrawView5.this.mPlayAnimator = ValueAnimator.ofInt(i5, length + i5);
                LineDrawView5.this.mPlayAnimator.setInterpolator(new LinearInterpolator());
                LineDrawView5.this.mPlayAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.surface_view.LineDrawView5.1.1
                    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                    public void onAnimationUpdate(ValueAnimator valueAnimator2) {
                        LineDrawView5.this.mPlayTime = ((Integer) valueAnimator2.getAnimatedValue()).intValue();
                    }
                });
                LineDrawView5.this.mPlayAnimator.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.surface_view.LineDrawView5.1.2
                    @Override // android.animation.Animator.AnimatorListener
                    public void onAnimationCancel(Animator animator) {
                    }

                    @Override // android.animation.Animator.AnimatorListener
                    public void onAnimationEnd(Animator animator) {
                        LineDrawView5.this.mPlayTime = i5 + length;
                    }

                    @Override // android.animation.Animator.AnimatorListener
                    public void onAnimationRepeat(Animator animator) {
                    }

                    @Override // android.animation.Animator.AnimatorListener
                    public void onAnimationStart(Animator animator) {
                    }
                });
                LineDrawView5.this.mPlayAnimator.setDuration(500L);
                LineDrawView5.this.mPlayAnimator.start();
            }
        }, 100L);
        this.mHandler.sendEmptyMessage(1);
        sendStatus();
    }

    public void addFunData(int i, int i2) {
        if (this.isRecord) {
            String upperCase = Integer.toHexString(i).toUpperCase();
            if (upperCase.length() < 2) {
                upperCase = "0" + upperCase;
            }
            String upperCase2 = Integer.toHexString(i2).toUpperCase();
            if (upperCase2.length() < 2) {
                upperCase2 = "0" + upperCase2;
            }
            this.funDataMap.put(Integer.valueOf(this.mTime), upperCase + upperCase2);
            sendFunData(upperCase + upperCase2);
        }
    }

    public void copy() {
        if (this.isSelect) {
            this.mCopyData = getSelectData(this.strDataBuffer, this.mSelectStartTime, this.mSelectEndTime);
            this.mCopyVibrateData = getSelectData(this.strVibrateDataBuffer, this.mSelectStartTime, this.mSelectEndTime);
            this.mCopyStartPositionData = getSelectData(this.strStartPositionDataBuffer, this.mSelectStartTime, this.mSelectEndTime);
            this.mCopySmoothData = getSelectData(this.strSmoothDataBuffer, this.mSelectStartTime, this.mSelectEndTime);
            this.mCopyFunDataMap.clear();
            int i = this.mSelectStartTime;
            int i2 = this.intervalTime;
            int i3 = (i / i2) * i2;
            int i4 = (this.mSelectEndTime / i2) * i2;
            for (Map.Entry<Integer, String> entry : this.funDataMap.entrySet()) {
                int intValue = entry.getKey().intValue();
                if (intValue >= i3 && intValue <= i4) {
                    this.mCopyFunDataMap.put(Integer.valueOf(intValue - i3), entry.getValue());
                }
            }
            LogUtil.e("copyFunDataMap size：" + this.mCopyFunDataMap.size());
            select(false);
            sendStatus();
        }
    }

    public void delete() {
        try {
            if (this.isSelect) {
                int i = this.mSelectStartTime;
                int i2 = this.intervalTime;
                int i3 = (i / i2) * 3;
                int i4 = ((this.mSelectEndTime / i2) + 1) * 3;
                if (i4 > this.strDataBuffer.length()) {
                    i4 = this.strDataBuffer.length();
                }
                this.strDataBuffer.delete(i3, i4);
                this.strVibrateDataBuffer.delete(i3, i4);
                this.strStartPositionDataBuffer.delete(i3, i4);
                this.strSmoothDataBuffer.delete(i3, i4);
                removeFunDataInRange(this.mSelectStartTime, this.mSelectEndTime);
                LogUtil.e("strData = " + ((Object) this.strDataBuffer));
                LogUtil.e("stringBuilder  = " + this.strDataBuffer.toString());
                int length = ((this.strDataBuffer.length() / 3) - 1) * this.intervalTime;
                this.mTime = length;
                if (this.mPlayTime > length) {
                    this.mPlayTime = length;
                }
                this.mHandler.sendEmptyMessage(1);
                setSelect(false);
                sendStatus();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public int getIntervalTime() {
        return this.intervalTime;
    }

    public int getIsPause() {
        return this.isPause;
    }

    public String getSelectData(int i, int i2) {
        String str = "";
        try {
            int i3 = this.intervalTime;
            int i4 = (i / i3) * 3;
            int i5 = ((i2 / i3) + 1) * 3;
            if (i5 > this.strDataBuffer.length()) {
                i5 = this.strDataBuffer.length();
                LogUtil.e("超长");
            }
            LogUtil.e("mTime：" + this.mTime + ", mPlayTime：" + this.mPlayTime);
            LogUtil.e("startTime：" + i + ", endTime：" + i2);
            LogUtil.e("startIndex：" + i4 + ", endIndex：" + i5);
            LogUtil.e("strData.length()：" + this.strDataBuffer.length());
            str = this.strDataBuffer.substring(i4, i5);
            LogUtil.e(new StringBuilder("strDataBuffer：").append(this.strDataBuffer.substring(r10.length() - 1500)).toString());
            LogUtil.e("copyData：" + str);
            return str;
        } catch (Exception e) {
            e.printStackTrace();
            return str;
        }
    }

    public String getSelectData(StringBuffer stringBuffer, int i, int i2) {
        String str = "";
        try {
            int i3 = this.intervalTime;
            int i4 = (i / i3) * 3;
            int i5 = ((i2 / i3) + 1) * 3;
            if (i5 > stringBuffer.length()) {
                i5 = stringBuffer.length();
                LogUtil.e("超长");
            }
            LogUtil.e("mTime：" + this.mTime + ", mPlayTime：" + this.mPlayTime);
            LogUtil.e("startTime：" + i + ", endTime：" + i2);
            LogUtil.e("startIndex：" + i4 + ", endIndex：" + i5);
            LogUtil.e("strData.length()：" + stringBuffer.length());
            str = stringBuffer.substring(i4, i5);
            LogUtil.e("data：" + stringBuffer.substring(stringBuffer.length() - 1500));
            LogUtil.e("copyData：" + str);
            return str;
        } catch (Exception e) {
            e.printStackTrace();
            return str;
        }
    }

    public String getStrData() {
        StringBuilder sb = new StringBuilder(this.strDataBuffer.toString());
        LogUtil.e("getStrData 默认数据：" + sb.toString());
        int i = 0;
        for (Integer num : this.funDataMap.keySet()) {
            int intValue = (((num.intValue() / this.intervalTime) + 1) * 3) + i;
            LogUtil.e("getStrData funKey：" + num + ", 插入坐标：" + intValue + ", 插入数据：" + this.funDataMap.get(num) + this.mSeparator);
            String str = this.funDataMap.get(num) + this.mSeparator;
            try {
                sb.insert(intValue, str);
            } catch (Exception e) {
                e.printStackTrace();
                LogUtil.e("getStrData 插入失败，timeIndex：" + intValue + ", length:" + sb.length());
            }
            i += str.length();
        }
        LogUtil.e("getStrData 最终数据：" + sb.toString());
        return sb.substring(3);
    }

    public String getStrSmoothData() {
        return this.strSmoothDataBuffer.substring(3);
    }

    public String getStrStartPositionData() {
        return this.strStartPositionDataBuffer.substring(3);
    }

    public String getStrVibrateData() {
        StringBuilder sb = new StringBuilder(this.strVibrateDataBuffer.toString());
        int i = 0;
        for (Integer num : this.funVibrateDataMap.keySet()) {
            if (this.funVibrateDataMap.get(num) != null) {
                int intValue = (((num.intValue() / this.intervalTime) + 1) * 3) + i;
                String str = this.funVibrateDataMap.get(num) + this.mSeparator;
                sb.insert(intValue, str);
                i += str.length();
            }
        }
        return sb.substring(3);
    }

    public int getmPlayTime() {
        return this.mPlayTime;
    }

    public int getmTime() {
        return this.mTime;
    }

    public boolean isCopy() {
        return !TextUtils.isEmpty(this.mCopyData);
    }

    public boolean isDataEmpty() {
        StringBuffer stringBuffer = this.strDataBuffer;
        return stringBuffer == null || stringBuffer.length() <= 3;
    }

    public boolean isDataEmpty(StringBuffer stringBuffer) {
        return stringBuffer == null || stringBuffer.length() <= 3;
    }

    public boolean isDrawPiu() {
        return this.isDrawPiu;
    }

    public boolean isPiuDragRemoveEnabled() {
        return this.isPiuDragRemoveEnabled;
    }

    public boolean isPlay() {
        return this.isPlay;
    }

    public boolean isPlayCircle() {
        return this.isPlayCircle;
    }

    public boolean isRecord() {
        return this.isRecord;
    }

    public boolean isSelect() {
        return this.isSelect;
    }

    public boolean isSelectList() {
        return this.isSelectList;
    }

    public boolean isSmoothLine() {
        return this.isSmoothLine;
    }

    public boolean isStartPositionLine() {
        return this.isStartPositionLine;
    }

    public boolean isVibrateLine() {
        return this.isVibrateLine;
    }

    @Override // android.view.SurfaceView, android.view.View
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        LogUtil.e("窗口onAttachedToWindow()");
        if (this.isPrepare) {
            return;
        }
        LogUtil.e("窗口线程()");
        this.isPrepare = true;
        Thread thread = new Thread(this);
        this.mThread = thread;
        thread.start();
    }

    @Override // android.view.SurfaceView, android.view.View
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        LogUtil.e("窗口onDetachedFromWindow()");
        this.isPrepare = false;
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        int action = motionEvent.getAction() & 255;
        int x = (int) motionEvent.getX();
        int y = (int) motionEvent.getY();
        if (action != 0) {
            if (action != 1) {
                if (action != 2) {
                    if (action != 3) {
                        if (action != 5) {
                            if (action == 6 && this.mPiuTouchState != 1) {
                                this.needDelay = true;
                                this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.surface_view.LineDrawView5.4
                                    @Override // java.lang.Runnable
                                    public void run() {
                                        LineDrawView5.this.needDelay = false;
                                    }
                                }, 200L);
                            }
                        } else if (this.mPiuTouchState != 1 && motionEvent.getPointerCount() > 1) {
                            float x2 = motionEvent.getX(1);
                            this.downTwoX = x2;
                            this.moveCenterX = (this.downOneX + x2) / 2.0f;
                            LogUtil.e("moveCenterX：" + this.moveCenterX);
                        }
                    }
                } else if (this.mPiuTouchState != 1 || this.mDragItem == null) {
                    move(motionEvent);
                } else {
                    movePiuItem(x, y);
                }
            }
            if (this.mPiuTouchState != 1 || this.mDragItem == null) {
                this.needDelay = false;
                this.inRectTarget = 0;
                this.isStartSelectAnim = false;
                this.isStartSelectAnimMoveTime = 0.0f;
                this.isEndSelectAnim = false;
                this.isEndSelectAnimMoveTime = 0.0f;
            } else {
                releasePiuItem();
            }
        } else {
            this.downOneX = motionEvent.getX(0);
            if (this.isRecord || this.isPlay || this.isSelect || this.isSelectList || !grabPiuItem(x, y)) {
                float f = x;
                float f2 = y;
                if (inInTarget(this.mLeftRectF, f, f2)) {
                    this.needDelay = true;
                    this.inRectTarget++;
                } else if (inInTarget(this.mRightRectF, f, f2)) {
                    this.needDelay = true;
                    this.inRectTarget += 2;
                }
            } else {
                this.needDelay = false;
                this.inRectTarget = 0;
                LogUtil.e("PIU拖拽开始 actualTime：" + this.mDragItem.actualTime);
            }
        }
        return true;
    }

    public void paste() {
        try {
            if (isCopy()) {
                if (!this.isSelect) {
                    if (this.isPause == 2) {
                        int i = this.mPlayTime;
                        int i2 = this.intervalTime;
                        remapFunDataMapAfterPaste(((i / i2) + 1) * i2, 0, this.mCopyData.length() / 3);
                        addData(this.mCopyData, this.mCopyVibrateData, this.mCopyStartPositionData, this.mCopySmoothData);
                        return;
                    }
                    return;
                }
                int i3 = this.mSelectStartTime;
                int i4 = this.intervalTime;
                int i5 = (i3 / i4) * 3;
                int i6 = ((this.mSelectEndTime / i4) + 1) * 3;
                if (i6 > this.strDataBuffer.length()) {
                    i6 = this.strDataBuffer.length();
                }
                this.strDataBuffer.replace(i5, i6, this.mCopyData);
                this.strVibrateDataBuffer.replace(i5, i6, this.mCopyVibrateData);
                this.strStartPositionDataBuffer.replace(i5, i6, this.mCopyStartPositionData);
                this.strSmoothDataBuffer.replace(i5, i6, this.mCopySmoothData);
                remapFunDataMapAfterPaste((i5 / 3) * this.intervalTime, (i6 - i5) / 3, this.mCopyData.length() / 3);
                this.mSelectEndTime = this.mSelectStartTime + (((this.mCopyData.length() / 3) - 1) * this.intervalTime);
                int length = ((this.strDataBuffer.length() / 3) - 1) * this.intervalTime;
                this.mTime = length;
                if (this.mPlayTime > length) {
                    this.mPlayTime = length;
                }
                this.mHandler.sendEmptyMessage(1);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void pause() {
        if (this.isRecord) {
            this.isRecord = false;
            this.isPause = 1;
            sendStatus();
        }
        if (this.isPlay) {
            this.isPlay = false;
            this.isPause = 2;
            sendStatus();
        }
    }

    public void play() {
        if (this.isPlay) {
            return;
        }
        this.isRecord = false;
        this.isPlay = true;
        this.isPause = 0;
        this.isSelectList = false;
        if (this.mPlayTime >= this.mTime) {
            this.mPlayTime = 0;
        }
        sendStatus();
    }

    public void recode() {
        if (this.isRecord) {
            return;
        }
        this.isRecord = true;
        this.isPause = 0;
        this.isPlay = false;
        this.isSelectList = false;
        sendStatus();
    }

    @Override // java.lang.Runnable
    public void run() {
        while (this.isPrepare) {
            long currentTimeMillis = System.currentTimeMillis();
            draw();
            long currentTimeMillis2 = System.currentTimeMillis();
            long j = currentTimeMillis2 - currentTimeMillis;
            if (this.high < j) {
                this.high = j;
            }
            this.lastEnd = currentTimeMillis2;
            try {
                int i = this.intervalTime;
                if (j < i) {
                    Thread.sleep(i - j);
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public void select(boolean z) {
        this.isSelect = z;
        if (z) {
            int i = this.mShowTime;
            this.mSelectStartTime = i;
            int i2 = this.mTime;
            int i3 = this.mScreenTime;
            if (i2 < i + i3) {
                this.mSelectEndTime = i2;
            } else {
                i2 = i + i3;
                this.mSelectEndTime = i2;
            }
            OnSelectTimeListener onSelectTimeListener = this.onSelectTimeListener;
            if (onSelectTimeListener != null) {
                onSelectTimeListener.onSelectTime(i, i2);
            }
        }
        sendStatus();
    }

    public void selectList(boolean z) {
        this.isSelectList = z;
        if (z) {
            if (this.isRecord || this.isPause == 1) {
                this.isRecord = false;
                this.mPlayTime = this.mTime;
            }
            if (this.isPlay) {
                this.isPlay = false;
            }
            if (this.isSelect) {
                this.isSelect = false;
            }
            this.isPause = 2;
        }
        sendStatus();
    }

    public void setData(String str, String str2, String str3, String str4) {
        if (TextUtils.isEmpty(str3) || TextUtils.isEmpty(str4)) {
            this.mType = 1;
        } else {
            this.mType = 2;
        }
        this.isPause = 0;
        this.strDataBuffer = new StringBuffer("00" + this.mSeparator + str);
        this.strVibrateDataBuffer = new StringBuffer("00" + this.mSeparator + str2);
        this.strStartPositionDataBuffer = new StringBuffer("00" + this.mSeparator + str3);
        this.strSmoothDataBuffer = new StringBuffer("00" + this.mSeparator + str4);
        initFunData();
        replenishData(this.strVibrateDataBuffer);
        replenishData(this.strStartPositionDataBuffer);
        replenishData(this.strSmoothDataBuffer);
        this.mTime = ((this.strDataBuffer.length() / 3) - 1) * this.intervalTime;
        this.mHandler.sendEmptyMessage(1);
        sendStatus();
    }

    public void setDrawPiu(boolean z) {
        this.isDrawPiu = z;
    }

    public void setDrawYScale(boolean z) {
        this.isDrawYScale = z;
        if (z) {
            this.mScaleLeft = (int) (MeasureUtil.dip2px(getContext(), 1.0f) + MeasureUtil.calcTextWidth(this.mTextPaint, "100") + this.mLineTextIntersticeWidth + (this.mSelectLineWidth / 2));
        } else {
            this.mScaleLeft = MeasureUtil.dip2px(getContext(), 1.0f);
        }
    }

    public void setHigh(int i) {
        this.high = i;
    }

    public void setIntervalTime(int i) {
        this.intervalTime = i;
    }

    public void setOnPlayTimeListener(OnPlayTimeListener onPlayTimeListener) {
        this.onPlayTimeListener = onPlayTimeListener;
    }

    public void setOnRecodeTimeListener(OnRecodeTimeListener onRecodeTimeListener) {
        this.onRecodeTimeListener = onRecodeTimeListener;
    }

    public void setOnSelectTimeListener(OnSelectTimeListener onSelectTimeListener) {
        this.onSelectTimeListener = onSelectTimeListener;
    }

    public void setOnStatusChangeListener(OnStatusChangeListener onStatusChangeListener) {
        this.onStatusChangeListener = onStatusChangeListener;
    }

    public void setPiuDragRemoveEnabled(boolean z) {
        this.isPiuDragRemoveEnabled = z;
    }

    public void setPlay(boolean z) {
        this.isPlay = z;
    }

    public void setPlayCircle(boolean z) {
        this.isPlayCircle = z;
    }

    public void setPlayTime(int i) {
        this.mPlayTime = i;
        int i2 = this.mTime;
        if (i > i2) {
            this.mPlayTime = i2;
        }
        this.isPause = 2;
    }

    public void setRecord(boolean z) {
        this.isRecord = z;
    }

    public void setSelect(boolean z) {
        this.isSelect = z;
    }

    public void setSmoothLine(boolean z) {
        this.isSmoothLine = z;
    }

    public void setStartPositionLine(boolean z) {
        this.isStartPositionLine = z;
    }

    public void setVibrateLine(boolean z) {
        this.isVibrateLine = z;
    }

    public void setmCurrentScale(int i) {
        this.mCurrentScale = i;
    }

    public void setmCurrentVibrateScale(int i) {
        this.mCurrentVibrateScale = i;
    }

    public void setmSmoothScale(int i) {
        this.mSmoothScale = i;
    }

    public void setmStartPositionScale(int i) {
        this.mStartPositionScale = i;
    }

    public void setmType(int i) {
        this.mType = i;
    }

    @Override // android.view.SurfaceHolder.Callback
    public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i2, int i3) {
        LinearGradient linearGradient;
        this.mWidth = getWidth();
        int height = getHeight();
        this.mHeight = height;
        this.mScaleWidth = (this.mWidth - this.mScaleLeft) - this.mScaleRight;
        this.mScaleHeight = (height - this.mScaleTop) - this.mScaleBottom;
        int i4 = 0;
        while (true) {
            Paint[] paintArr = this.mPaints;
            if (i4 >= paintArr.length) {
                break;
            }
            if (paintArr[i4] != null) {
                LinearGradient linearGradient2 = new LinearGradient(0.0f, 0.0f, this.mScaleWidth, 0.0f, new int[]{getResources().getColor(R.color.play_mode_line_left), getResources().getColor(R.color.play_mode_line_right)}, new float[]{0.0f, 1.0f}, Shader.TileMode.CLAMP);
                if (i4 == 1) {
                    linearGradient = new LinearGradient(0.0f, 0.0f, this.mScaleWidth, 0.0f, new int[]{getResources().getColor(R.color.play_mode_vibrate_line_left), getResources().getColor(R.color.play_mode_vibrate_line_right)}, new float[]{0.0f, 1.0f}, Shader.TileMode.CLAMP);
                } else {
                    if (i4 == 2) {
                        linearGradient2 = new LinearGradient(0.0f, 0.0f, this.mScaleWidth, 0.0f, new int[]{getResources().getColor(R.color.play_mode_start_position_line_left), getResources().getColor(R.color.play_mode_start_position_line_right)}, new float[]{0.0f, 1.0f}, Shader.TileMode.CLAMP);
                    } else if (i4 == 3) {
                        linearGradient = new LinearGradient(0.0f, 0.0f, this.mScaleWidth, 0.0f, new int[]{getResources().getColor(R.color.play_mode_smooth_line_left), getResources().getColor(R.color.play_mode_smooth_line_right)}, new float[]{0.0f, 1.0f}, Shader.TileMode.CLAMP);
                    }
                    this.mPaints[i4].setShader(linearGradient2);
                }
                linearGradient2 = linearGradient;
                this.mPaints[i4].setShader(linearGradient2);
            }
            i4++;
        }
        this.mScaleTopY = this.mScaleTop;
        this.mScaleBottomY = this.mHeight - this.mScaleBottom;
        this.mScaleLeftX = this.mScaleLeft;
        this.mScaleRightX = this.mWidth - this.mScaleRight;
        float f = (this.mScaleHeight - (this.mXScaleLongLine * 2)) - this.mTextHeight;
        this.mYHeight = f;
        this.mYItemHeight = f / this.mYItemNum;
        int i5 = this.mScaleWidth;
        int i6 = this.mScaleShortNum;
        this.mMaxScaleItemWidth = ((i5 * 1.0f) / i6) / 2.0f;
        this.mMinScaleItemWidth = ((i5 * 1.0f) / i6) / 60.0f;
        LogUtil.e("mMaxScaleItemWidth：" + this.mMaxScaleItemWidth + ", mMinScaleItemWidth：" + this.mMinScaleItemWidth);
        float f2 = this.mScaleItemWidth;
        if (f2 == 0.0f) {
            f2 = ((this.mScaleWidth * 1.0f) / this.mScaleShortNum) / 8.7f;
            this.mScaleItemWidth = f2;
        }
        int i7 = this.mScaleWidth;
        int i8 = this.mScaleShortNum;
        this.mScaleNum = (int) ((i7 / f2) + (i8 * 2));
        int i9 = this.intervalTime;
        float f3 = (f2 * i8) / (1000.0f / i9);
        this.mDistance = f3;
        this.mScreenTime = (int) ((i7 / f3) * i9);
        this.mThumbTime = (int) ((this.mThumbWidth / f3) * i9);
    }

    @Override // android.view.SurfaceHolder.Callback
    public void surfaceCreated(SurfaceHolder surfaceHolder) {
        this.isRunning = true;
        Choreographer.getInstance().postFrameCallback(new Choreographer.FrameCallback() { // from class: com.hismith.hismithapp.surface_view.LineDrawView5.3
            @Override // android.view.Choreographer.FrameCallback
            public void doFrame(long j) {
                LogUtil.e("doFrame（）" + j);
            }
        });
    }

    @Override // android.view.SurfaceHolder.Callback
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {
        this.isRunning = false;
    }
}
