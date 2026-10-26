package com.hismith.hismithapp.surface_view;

import android.animation.Animator;
import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.DashPathEffect;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
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
import java.util.LinkedHashMap;
import java.util.Map;

/* loaded from: classes3.dex */
public class LineDrawView6 extends SurfaceView implements SurfaceHolder.Callback, Runnable {
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
    private float distance;
    private float downOneX;
    private float downTwoX;
    private Map<Integer, String> funDataMap;
    private Map<Integer, String> funVibrateDataMap;
    private long high;
    private int inRectTarget;
    private int intervalTime;
    private boolean isDrawYScale;
    private boolean isEndSelectAnim;
    private float isEndSelectAnimMoveTime;
    private int isPause;
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
    private String mCopySmoothData;
    private String mCopyStartPositionData;
    private String mCopyVibrateData;
    private int mCurrentScale;
    private int mCurrentVibrateScale;
    float mCursorBorderRadius;
    private Paint mCursorPaint;
    private float mDistance;
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
    private StringBuffer strDataBuffer;
    private StringBuffer strSmoothDataBuffer;
    private StringBuffer strStartPositionDataBuffer;
    private StringBuffer strVibrateDataBuffer;
    int whiteHeight;
    int whiteWidth;

    public LineDrawView6(Context context) {
        this(context, null);
    }

    public LineDrawView6(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public LineDrawView6(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.isPrepare = false;
        this.isPause = 0;
        this.isSetData = false;
        this.isPlay = false;
        this.isSelect = false;
        this.isSelectList = false;
        this.isPlayCircle = false;
        this.isDrawYScale = true;
        this.mThumbWidth = MeasureUtil.dip2px(getContext(), 15.0f);
        this.mScaleTop = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mScaleBottom = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mScaleLeft = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mScaleRight = MeasureUtil.dip2px(getContext(), 1.0f);
        this.bgColor = getResources().getColor(R.color.preview2_bgcolor);
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
        this.intervalTime = 25;
        this.mCurrentScale = 0;
        this.mCurrentVibrateScale = 0;
        this.mStartPositionScale = 0;
        this.mSmoothScale = 0;
        this.isVibrateLine = false;
        this.isStartPositionLine = false;
        this.isSmoothLine = false;
        this.mType = 1;
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
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.surface_view.LineDrawView6.2
            /* JADX WARN: Can't fix incorrect switch cases order, some code will duplicate */
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                int i2 = message.what;
                if (i2 != 16) {
                    switch (i2) {
                        case 1:
                            if (LineDrawView6.this.onRecodeTimeListener != null) {
                                LineDrawView6.this.onRecodeTimeListener.onRecodeTime(LineDrawView6.this.mTime);
                                break;
                            }
                            break;
                        case 2:
                            int intValue = ((Integer) message.obj).intValue();
                            if (LineDrawView6.this.onPlayTimeListener != null) {
                                LineDrawView6.this.onPlayTimeListener.onPlayTime(intValue);
                                break;
                            }
                            break;
                        case 3:
                            if (LineDrawView6.this.onStatusChangeListener != null) {
                                LineDrawView6.this.onStatusChangeListener.OnStatusChange();
                                break;
                            }
                            break;
                        case 4:
                            if (LineDrawView6.this.onPlayTimeListener != null) {
                                LineDrawView6.this.onPlayTimeListener.onComplete();
                                break;
                            }
                            break;
                        case 5:
                            int intValue2 = ((Integer) message.obj).intValue();
                            if (LineDrawView6.this.onPlayTimeListener != null) {
                                LineDrawView6.this.onPlayTimeListener.onSpeedChange(intValue2);
                                break;
                            }
                            break;
                        case 6:
                            int intValue3 = ((Integer) message.obj).intValue();
                            if (LineDrawView6.this.onPlayTimeListener != null) {
                                LineDrawView6.this.onPlayTimeListener.onVibrateChange(intValue3);
                                break;
                            }
                            break;
                        case 7:
                            if (LineDrawView6.this.onPlayTimeListener != null) {
                                LineDrawView6.this.onPlayTimeListener.onLinePause();
                                break;
                            }
                            break;
                        case 8:
                            String str = (String) message.obj;
                            if (!TextUtils.isEmpty(str) && str.length() > 2 && LineDrawView6.this.onPlayTimeListener != null) {
                                LineDrawView6.this.onPlayTimeListener.onFunData(str.substring(0, 2), str.substring(2));
                                break;
                            }
                            break;
                        case 9:
                            int i3 = message.arg1;
                            int i4 = message.arg2;
                            if (LineDrawView6.this.onPlayTimeListener != null) {
                                LineDrawView6.this.onPlayTimeListener.onPositionRangeChange(i3, i4);
                                break;
                            }
                            break;
                    }
                } else {
                    int intValue4 = ((Integer) message.obj).intValue();
                    if (LineDrawView6.this.onPlayTimeListener != null) {
                        LineDrawView6.this.onPlayTimeListener.onSmoothChange(intValue4);
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
        init();
    }

    /* JADX WARN: Removed duplicated region for block: B:112:0x0139  */
    /* JADX WARN: Removed duplicated region for block: B:126:0x0177  */
    /* JADX WARN: Removed duplicated region for block: B:57:0x0310 A[Catch: all -> 0x031f, Exception -> 0x0322, TRY_LEAVE, TryCatch #5 {Exception -> 0x0322, blocks: (B:24:0x01a7, B:26:0x01b1, B:28:0x01bf, B:29:0x01f6, B:31:0x01fa, B:32:0x0213, B:34:0x0217, B:35:0x0230, B:37:0x0253, B:39:0x0257, B:42:0x025c, B:44:0x0260, B:45:0x0269, B:47:0x0271, B:48:0x028e, B:50:0x0292, B:51:0x02af, B:53:0x02b3, B:54:0x02d0, B:55:0x02f0, B:57:0x0310), top: B:23:0x01a7, outer: #1 }] */
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
        try {
            if (!this.isRunning) {
                return;
            }
            try {
                try {
                    Canvas lockCanvas = this.mHolder.lockCanvas();
                    this.mCanvas = lockCanvas;
                    if (lockCanvas != null) {
                        lockCanvas.drawColor(this.bgColor);
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
            } finally {
            }
        } catch (Exception e2) {
            e2.printStackTrace();
        }
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

    /* JADX WARN: Multi-variable type inference failed */
    /* JADX WARN: Removed duplicated region for block: B:20:0x00a7  */
    /* JADX WARN: Removed duplicated region for block: B:33:0x0122  */
    /* JADX WARN: Removed duplicated region for block: B:36:0x0135 A[ADDED_TO_REGION] */
    /* JADX WARN: Removed duplicated region for block: B:39:0x013d  */
    /* JADX WARN: Removed duplicated region for block: B:40:0x0141  */
    /* JADX WARN: Removed duplicated region for block: B:42:0x0146  */
    /* JADX WARN: Removed duplicated region for block: B:57:0x018e A[ADDED_TO_REGION, SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:58:0x0126  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private void drawLine(Canvas canvas, StringBuffer stringBuffer, Path path, Path path2, Paint paint, Paint paint2, int i, float f, boolean z, Map<Integer, String> map) {
        StringBuffer stringBuffer2;
        int i2;
        Object[] objArr;
        int i3;
        float f2;
        int i4;
        String str;
        if (canvas != null) {
            path.reset();
            path2.reset();
            ArrayList<float[]> arrayList = new ArrayList();
            int i5 = this.intervalTime;
            int i6 = i % i5;
            float f3 = ((i6 * 1.0f) / i5) * this.mDistance;
            int i7 = i / i5;
            int i8 = i7 * 3;
            int i9 = i7 * i5;
            boolean z2 = true;
            if (stringBuffer.length() > i8) {
                int ceil = ((((int) Math.ceil(((this.mScreenTime + i6) * 1.0f) / this.intervalTime)) + 1) * 3) + i8;
                stringBuffer2 = ceil <= stringBuffer.length() ? new StringBuffer(stringBuffer.substring(i8, ceil)) : new StringBuffer(stringBuffer.substring(i8));
            } else {
                stringBuffer2 = null;
            }
            int i10 = 0;
            while (stringBuffer2 != null && i10 < stringBuffer2.length() / 3) {
                float f4 = ((this.mDistance * i10) - f3) + this.mScaleLeftX;
                int i11 = i10 * 3;
                try {
                    i2 = Integer.valueOf(stringBuffer2.substring(i11, i11 + 2), 16).intValue();
                } catch (Exception unused) {
                    i2 = 0;
                }
                int i12 = this.mScaleLeftX;
                boolean z3 = z2;
                if (f4 < i12) {
                    f4 = i12;
                } else {
                    int i13 = this.mScaleRightX;
                    if (f4 > i13) {
                        f4 = i13;
                    } else {
                        objArr = false;
                        if (objArr != false) {
                            if (i10 < (stringBuffer2.length() / 3) - 1) {
                                int i14 = (i10 + 1) * 3;
                                int intValue = Integer.valueOf(stringBuffer2.substring(i14, i14 + 2), 16).intValue();
                                int abs = Math.abs(intValue - i2);
                                int i15 = this.intervalTime;
                                int i16 = (abs * (i % i15)) / i15;
                                i2 = intValue > i2 ? i2 + i16 : i2 - i16;
                            } else if (i10 == (stringBuffer2.length() / 3) - 1) {
                                int i17 = (i10 - 1) * 3;
                                int intValue2 = Integer.valueOf(stringBuffer2.substring(i17, i17 + 2), 16).intValue();
                                int abs2 = Math.abs(intValue2 - i2);
                                int i18 = this.mScreenTime + i6;
                                i3 = i10;
                                int i19 = this.intervalTime;
                                int i20 = (abs2 * (i18 % i19)) / i19;
                                i2 = intValue2 > i2 ? intValue2 - i20 : intValue2 + i20;
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
                                if (map == null && !map.isEmpty()) {
                                    i4 = (this.intervalTime * i3) + i9;
                                    Integer.toHexString(255).toUpperCase();
                                    String upperCase = Integer.toHexString(Constans.START_TYPE_PIU).toUpperCase();
                                    if (map.containsKey(Integer.valueOf(i4)) && (str = map.get(Integer.valueOf(i4))) != null && str.toUpperCase().startsWith(upperCase)) {
                                        float[] fArr = new float[2];
                                        fArr[0] = f4;
                                        fArr[z3 ? 1 : 0] = f2;
                                        arrayList.add(fArr);
                                    }
                                }
                                i10 = i3 + 1;
                                z2 = z3 ? 1 : 0;
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
                        if (map == null) {
                            i4 = (this.intervalTime * i3) + i9;
                            Integer.toHexString(255).toUpperCase();
                            String upperCase2 = Integer.toHexString(Constans.START_TYPE_PIU).toUpperCase();
                            if (map.containsKey(Integer.valueOf(i4))) {
                                float[] fArr2 = new float[2];
                                fArr2[0] = f4;
                                fArr2[z3 ? 1 : 0] = f2;
                                arrayList.add(fArr2);
                            }
                        }
                        i10 = i3 + 1;
                        z2 = z3 ? 1 : 0;
                    }
                }
                objArr = z3 ? 1 : 0;
                if (objArr != false) {
                }
                i3 = i10;
                float f522 = this.mYHeight;
                f2 = (f522 - ((f522 / 100.0f) * i2)) + ((this.mHeight - f522) / 2.0f);
                if (path.isEmpty()) {
                }
                if (f - f4 >= -1.0E-4d) {
                }
                if (map == null) {
                }
                i10 = i3 + 1;
                z2 = z3 ? 1 : 0;
            }
            boolean z4 = z2;
            canvas.drawPath(path, paint);
            canvas.drawPath(path2, paint2);
            Bitmap bitmap = this.mFunPiuIcon;
            if (bitmap == null || bitmap.isRecycled()) {
                return;
            }
            for (float[] fArr3 : arrayList) {
                float f6 = fArr3[0];
                float f7 = fArr3[z4 ? 1 : 0];
                canvas.drawBitmap(this.mFunPiuIcon, f6 - (r5.getWidth() / 2.0f), f7 - (this.mFunPiuIcon.getHeight() / 2.0f), this.mFunPaint);
            }
        }
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
        this.mHolder.setFormat(-3);
        Paint paint = new Paint();
        this.mTextPaint = paint;
        paint.setAntiAlias(true);
        this.mTextPaint.setStyle(Paint.Style.FILL);
        this.mTextPaint.setTextSize(this.mTextSize);
        this.mTextPaint.setColor(Color.parseColor("#7FFFFFFF"));
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
        this.mCursorPaint.setColor(Color.parseColor("#33FFFFFF"));
        Paint paint4 = new Paint();
        this.mSelectPaint = paint4;
        paint4.setAntiAlias(true);
        this.mSelectPaint.setStyle(Paint.Style.STROKE);
        this.mSelectPaint.setColor(Color.parseColor("#FFFFFF"));
        this.mSelectPaint.setStrokeWidth(this.mStrokeWidth);
        this.mSelectPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mSelectPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint5 = new Paint();
        this.mDullPaint = paint5;
        paint5.setAntiAlias(true);
        this.mDullPaint.setStyle(Paint.Style.FILL);
        this.mDullPaint.setColor(Color.parseColor("#99000000"));
        Paint paint6 = new Paint();
        this.mEffectPaint = paint6;
        paint6.setAntiAlias(true);
        this.mEffectPaint.setStyle(Paint.Style.STROKE);
        this.mEffectPaint.setStrokeWidth(this.mStrokeWidth);
        this.mEffectPaint.setColor(Color.parseColor("#18FFFFFF"));
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
                this.mFunPiuIcon = Bitmap.createScaledBitmap(decodeResource, i2, i2, true);
                Paint paint7 = new Paint();
                this.mFunPaint = paint7;
                paint7.setAntiAlias(true);
                this.mFunPaint.setFilterBitmap(true);
                this.mFunPaint.setAlpha(255);
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

    /* JADX WARN: Removed duplicated region for block: B:11:0x0041 A[Catch: all -> 0x0171, TryCatch #0 {, blocks: (B:4:0x0005, B:6:0x000f, B:11:0x0041, B:12:0x007b, B:16:0x0036, B:19:0x008b, B:21:0x008f, B:23:0x0093, B:25:0x0097, B:27:0x009b, B:31:0x00a3, B:32:0x00b5, B:34:0x00b9, B:36:0x00d5, B:38:0x00df, B:39:0x00e2, B:41:0x00e6, B:42:0x00e9, B:44:0x00ed, B:45:0x0105, B:46:0x016d, B:47:0x00f2, B:49:0x00fa, B:50:0x0100, B:51:0x0138, B:53:0x013c, B:55:0x0148, B:56:0x014b, B:58:0x014f, B:59:0x0152, B:61:0x015a, B:62:0x015f, B:64:0x0164, B:65:0x0169), top: B:3:0x0005 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private synchronized void move(MotionEvent motionEvent) {
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

    private void sendPause() {
        Message message = new Message();
        message.what = 7;
        this.mHandler.sendMessage(message);
        this.lastSpeed = 0;
        this.lastVibrate = 0;
    }

    private void sendPlayFunData() {
        Map<Integer, String> map = this.funDataMap;
        if (map == null || !map.containsKey(Integer.valueOf(this.mPlayTime))) {
            return;
        }
        String str = this.funDataMap.get(Integer.valueOf(this.mPlayTime));
        Message message = new Message();
        message.what = 8;
        message.obj = str;
        this.mHandler.sendMessageDelayed(message, 50L);
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
        this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.surface_view.LineDrawView6.1
            @Override // java.lang.Runnable
            public void run() {
                final int i5 = LineDrawView6.this.mPlayTime;
                LineDrawView6.this.mPlayAnimator = ValueAnimator.ofInt(i5, length + i5);
                LineDrawView6.this.mPlayAnimator.setInterpolator(new LinearInterpolator());
                LineDrawView6.this.mPlayAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.surface_view.LineDrawView6.1.1
                    @Override // android.animation.ValueAnimator.AnimatorUpdateListener
                    public void onAnimationUpdate(ValueAnimator valueAnimator2) {
                        LineDrawView6.this.mPlayTime = ((Integer) valueAnimator2.getAnimatedValue()).intValue();
                    }
                });
                LineDrawView6.this.mPlayAnimator.addListener(new Animator.AnimatorListener() { // from class: com.hismith.hismithapp.surface_view.LineDrawView6.1.2
                    @Override // android.animation.Animator.AnimatorListener
                    public void onAnimationCancel(Animator animator) {
                    }

                    @Override // android.animation.Animator.AnimatorListener
                    public void onAnimationEnd(Animator animator) {
                        LineDrawView6.this.mPlayTime = i5 + length;
                    }

                    @Override // android.animation.Animator.AnimatorListener
                    public void onAnimationRepeat(Animator animator) {
                    }

                    @Override // android.animation.Animator.AnimatorListener
                    public void onAnimationStart(Animator animator) {
                    }
                });
                LineDrawView6.this.mPlayAnimator.setDuration(500L);
                LineDrawView6.this.mPlayAnimator.start();
            }
        }, 100L);
        this.mHandler.sendEmptyMessage(1);
        sendStatus();
    }

    public void attachedToStartRun() {
        if (this.isPrepare) {
            return;
        }
        LogUtil.e("窗口线程()");
        this.isPrepare = true;
        Thread thread = new Thread(this);
        this.mThread = thread;
        thread.start();
    }

    public void copy() {
        if (this.isSelect) {
            this.mCopyData = getSelectData(this.strDataBuffer, this.mSelectStartTime, this.mSelectEndTime);
            this.mCopyVibrateData = getSelectData(this.strVibrateDataBuffer, this.mSelectStartTime, this.mSelectEndTime);
            this.mCopyStartPositionData = getSelectData(this.strStartPositionDataBuffer, this.mSelectStartTime, this.mSelectEndTime);
            this.mCopySmoothData = getSelectData(this.strSmoothDataBuffer, this.mSelectStartTime, this.mSelectEndTime);
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

    public void detachedToStopRun() {
        this.isPrepare = false;
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
        return this.strDataBuffer.substring(3);
    }

    public String getStrSmoothData() {
        return this.strSmoothDataBuffer.substring(3);
    }

    public String getStrStartPositionData() {
        return this.strStartPositionDataBuffer.substring(3);
    }

    public String getStrVibrateData() {
        return this.strVibrateDataBuffer.substring(3);
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
        attachedToStartRun();
    }

    @Override // android.view.SurfaceView, android.view.View
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        LogUtil.e("窗口onDetachedFromWindow()");
        detachedToStopRun();
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        int action = motionEvent.getAction() & 255;
        int x = (int) motionEvent.getX();
        int y = (int) motionEvent.getY();
        if (action != 0) {
            if (action != 1) {
                if (action == 2) {
                    move(motionEvent);
                } else if (action != 3) {
                    if (action != 5) {
                        if (action == 6) {
                            this.needDelay = true;
                            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.surface_view.LineDrawView6.4
                                @Override // java.lang.Runnable
                                public void run() {
                                    LineDrawView6.this.needDelay = false;
                                }
                            }, 200L);
                        }
                    } else if (motionEvent.getPointerCount() > 1) {
                        float x2 = motionEvent.getX(1);
                        this.downTwoX = x2;
                        this.moveCenterX = (this.downOneX + x2) / 2.0f;
                        LogUtil.e("moveCenterX：" + this.moveCenterX);
                    }
                }
            }
            this.needDelay = false;
            this.inRectTarget = 0;
            this.isStartSelectAnim = false;
            this.isStartSelectAnimMoveTime = 0.0f;
            this.isEndSelectAnim = false;
            this.isEndSelectAnimMoveTime = 0.0f;
        } else {
            this.downOneX = motionEvent.getX(0);
            float f = x;
            float f2 = y;
            if (inInTarget(this.mLeftRectF, f, f2)) {
                this.needDelay = true;
                this.inRectTarget++;
            } else if (inInTarget(this.mRightRectF, f, f2)) {
                this.needDelay = true;
                this.inRectTarget += 2;
            }
        }
        return true;
    }

    public void paste() {
        try {
            if (isCopy()) {
                if (!this.isSelect) {
                    if (this.isPause == 2) {
                        addData(this.mCopyData, this.mCopyVibrateData, this.mCopyStartPositionData, this.mCopySmoothData);
                        return;
                    }
                    return;
                }
                int i = this.mSelectStartTime;
                int i2 = this.intervalTime;
                int i3 = (i / i2) * 3;
                int i4 = ((this.mSelectEndTime / i2) + 1) * 3;
                if (i4 > this.strDataBuffer.length()) {
                    i4 = this.strDataBuffer.length();
                }
                this.strDataBuffer.replace(i3, i4, this.mCopyData);
                this.strVibrateDataBuffer.replace(i3, i4, this.mCopyVibrateData);
                this.strStartPositionDataBuffer.replace(i3, i4, this.mCopyStartPositionData);
                this.strSmoothDataBuffer.replace(i3, i4, this.mCopySmoothData);
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
        Choreographer.getInstance().postFrameCallback(new Choreographer.FrameCallback() { // from class: com.hismith.hismithapp.surface_view.LineDrawView6.3
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
