package com.hismith.hismithapp.view;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Shader;
import android.text.TextUtils;
import android.util.AttributeSet;
import android.view.View;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.MeasureUtil;

/* loaded from: classes3.dex */
public class LinePointsView extends View {
    private int intervalTime;
    private boolean isShadow;
    private boolean isStretch;
    private float mDistance;
    private int mHeight;
    private boolean mIsAppend;
    private Paint mPaint;
    private Path mPath;
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
    private String mSeparator;
    private float mShadowCircle;
    private float mShadowDy;
    private Paint mShadowPaint;
    private Path mShadowPath;
    private Path mSmoothPath;
    private Path mSmoothShadowPath;
    private Path mStartPositionPath;
    private Path mStartPositionShadowPath;
    private float mStrokeWidth;
    private Path mVibratePath;
    private Path mVibrateShadowPath;
    private int mWidth;
    private int mXScaleLongLine;
    private int mXScaleShortLine;
    private float mYHeight;
    private float mYItemHeight;
    private int mYItemNum;
    private float screenTime;
    private StringBuffer strDataBuffer;
    private StringBuffer strSmoothDataBuffer;
    private StringBuffer strStartPositionDataBuffer;
    private StringBuffer strVibrateDataBuffer;

    public LinePointsView(Context context) {
        this(context, null);
    }

    public LinePointsView(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public LinePointsView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.mScaleTop = MeasureUtil.dip2px(getContext(), 5.0f);
        this.mScaleBottom = MeasureUtil.dip2px(getContext(), 10.0f);
        this.mScaleLeft = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mScaleRight = MeasureUtil.dip2px(getContext(), 1.0f);
        this.mStrokeWidth = MeasureUtil.dip2px(getContext(), 2.0f);
        this.mXScaleLongLine = MeasureUtil.dip2px(getContext(), 10.0f);
        this.mXScaleShortLine = MeasureUtil.dip2px(getContext(), 5.0f);
        this.mShadowCircle = MeasureUtil.dip2px(getContext(), 6.0f);
        this.mShadowDy = MeasureUtil.dip2px(getContext(), 10.0f);
        this.mYItemNum = 10;
        this.mScaleShortNum = 2;
        this.mScaleNum = 0;
        this.mSeparator = "G";
        this.intervalTime = 100;
        this.screenTime = 30.0f;
        this.mIsAppend = true;
        this.isShadow = true;
        this.isStretch = false;
        initType(attributeSet);
        init();
    }

    private void caleScale() {
        int i;
        float f;
        int i2 = this.intervalTime;
        float length = ((this.strDataBuffer.length() / 3.0f) * i2) / 1000.0f;
        float f2 = this.screenTime;
        int i3 = this.mScaleWidth;
        if (length < f2) {
            i = this.mScaleShortNum;
            f = ((i3 * 1.0f) / i) / length;
            this.mScaleItemWidth = f;
        } else {
            i = this.mScaleShortNum;
            f = ((i3 * 1.0f) / i) / f2;
            this.mScaleItemWidth = f;
        }
        this.mScaleNum = (int) ((i3 / f) + (i * 2));
        float f3 = (f * i) / (1000.0f / i2);
        this.mDistance = f3;
        this.mScreenTime = (int) ((i3 / f3) * i2);
    }

    private void caleScale2() {
        StringBuffer stringBuffer;
        float dip2px = MeasureUtil.dip2px(getContext(), 5.5f);
        this.mScaleItemWidth = dip2px;
        int i = this.mScaleWidth;
        int i2 = this.mScaleShortNum;
        this.mScaleNum = (int) ((i / dip2px) + (i2 * 2));
        int i3 = this.intervalTime;
        float f = (dip2px * i2) / (1000.0f / i3);
        this.mDistance = f;
        this.mScreenTime = (int) ((i / f) * i3);
        if (!this.isStretch || (stringBuffer = this.strDataBuffer) == null || stringBuffer.length() <= 0) {
            return;
        }
        int i4 = this.intervalTime;
        float length = ((this.strDataBuffer.length() / 3.0f) * i4) / 1000.0f;
        if (length < this.mScreenTime) {
            int i5 = this.mScaleWidth;
            int i6 = this.mScaleShortNum;
            float f2 = ((i5 * 1.0f) / i6) / length;
            this.mScaleItemWidth = f2;
            this.mScaleNum = (int) ((i5 / f2) + (i6 * 2));
            float f3 = (f2 * i6) / (1000.0f / i4);
            this.mDistance = f3;
            this.mScreenTime = (int) ((i5 / f3) * i4);
        }
    }

    private void drawLine(Canvas canvas) {
        StringBuffer stringBuffer;
        LinePointsView linePointsView;
        Canvas canvas2;
        if (canvas == null || (stringBuffer = this.strDataBuffer) == null || stringBuffer.length() <= 0) {
            return;
        }
        if (TextUtils.isEmpty(this.strSmoothDataBuffer)) {
            linePointsView = this;
            canvas2 = canvas;
        } else {
            this.mSmoothPath.reset();
            this.mSmoothShadowPath.reset();
            initSpeedLine(canvas, this.strSmoothDataBuffer, this.mSmoothPath, this.mSmoothShadowPath, new int[]{getResources().getColor(R.color.play_mode_smooth_line_left), getResources().getColor(R.color.play_mode_smooth_line_right)});
            linePointsView = this;
            canvas2 = canvas;
        }
        if (!TextUtils.isEmpty(linePointsView.strStartPositionDataBuffer)) {
            linePointsView.mStartPositionPath.reset();
            linePointsView.mStartPositionShadowPath.reset();
            linePointsView.initSpeedLine(canvas2, linePointsView.strStartPositionDataBuffer, linePointsView.mStartPositionPath, linePointsView.mStartPositionShadowPath, new int[]{linePointsView.getResources().getColor(R.color.play_mode_start_position_line_left), linePointsView.getResources().getColor(R.color.play_mode_start_position_line_right)});
        }
        if (!TextUtils.isEmpty(linePointsView.strVibrateDataBuffer)) {
            linePointsView.mVibratePath.reset();
            linePointsView.mVibrateShadowPath.reset();
            linePointsView.initSpeedLine(canvas2, linePointsView.strVibrateDataBuffer, linePointsView.mVibratePath, linePointsView.mVibrateShadowPath, new int[]{linePointsView.getResources().getColor(R.color.play_mode_vibrate_line_left), linePointsView.getResources().getColor(R.color.play_mode_vibrate_line_right)});
        }
        linePointsView.mPath.reset();
        linePointsView.mShadowPath.reset();
        linePointsView.initSpeedLine(canvas2, linePointsView.strDataBuffer, linePointsView.mPath, linePointsView.mShadowPath, new int[]{linePointsView.getResources().getColor(R.color.play_mode_line_left), linePointsView.getResources().getColor(R.color.play_mode_line_right)});
    }

    private void drawScale(Canvas canvas) {
        if (canvas != null) {
            float f = this.mScaleLeftX - (this.mStrokeWidth / 2.0f);
            int i = this.mScaleBottomY;
            canvas.drawLine(f, i, this.mScaleRightX, i, this.mScalePaint);
            for (int i2 = 0; i2 <= this.mScaleNum; i2++) {
                float f2 = i2 * this.mScaleItemWidth;
                int i3 = this.mScaleLeftX;
                float f3 = f2 + i3;
                if (f3 >= i3 && f3 <= this.mScaleRightX) {
                    int i4 = i2 % this.mScaleShortNum;
                    int i5 = this.mScaleBottomY;
                    if (i4 == 0) {
                        canvas.drawLine(f3, i5, f3, i5 - this.mXScaleLongLine, this.mScalePaint);
                    } else {
                        canvas.drawLine(f3, i5, f3, i5 - this.mXScaleShortLine, this.mScalePaint);
                    }
                }
            }
        }
    }

    private void init() {
        Paint paint = new Paint();
        this.mScalePaint = paint;
        paint.setAntiAlias(true);
        this.mScalePaint.setStyle(Paint.Style.STROKE);
        this.mScalePaint.setStrokeWidth(this.mStrokeWidth);
        this.mScalePaint.setColor(-7829368);
        Paint paint2 = new Paint();
        this.mPaint = paint2;
        paint2.setAntiAlias(true);
        this.mPaint.setStyle(Paint.Style.STROKE);
        this.mPaint.setStrokeWidth(this.mStrokeWidth);
        this.mPaint.setColor(Color.parseColor("#A0388F"));
        this.mPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint3 = new Paint();
        this.mShadowPaint = paint3;
        paint3.setAntiAlias(true);
        this.mShadowPaint.setStyle(Paint.Style.STROKE);
        this.mShadowPaint.setStrokeWidth(this.mStrokeWidth);
        this.mShadowPaint.setColor(Color.parseColor("#00000000"));
        this.mShadowPaint.setShadowLayer(this.mShadowCircle, 0.0f, this.mShadowDy, Color.parseColor("#4C000000"));
        this.mShadowPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mShadowPaint.setStrokeCap(Paint.Cap.ROUND);
        this.mPath = new Path();
        this.mShadowPath = new Path();
        this.mVibratePath = new Path();
        this.mVibrateShadowPath = new Path();
        this.mStartPositionPath = new Path();
        this.mStartPositionShadowPath = new Path();
        this.mSmoothPath = new Path();
        this.mSmoothShadowPath = new Path();
    }

    private void initFunData() {
        try {
            String upperCase = Integer.toHexString(255).toUpperCase();
            String upperCase2 = Integer.toHexString(Constans.START_TYPE_PIU).toUpperCase();
            if (this.strDataBuffer.toString().contains(upperCase) || this.strDataBuffer.toString().contains(upperCase2)) {
                String[] split = this.strDataBuffer.toString().split(this.mSeparator);
                this.strDataBuffer = new StringBuffer();
                for (int i = 0; i < split.length; i++) {
                    if (!split[i].toUpperCase().startsWith(upperCase) && !split[i].toUpperCase().startsWith(upperCase2)) {
                        this.strDataBuffer.append(split[i]);
                        this.strDataBuffer.append(this.mSeparator);
                    }
                }
            }
            if (this.strVibrateDataBuffer.toString().contains(upperCase)) {
                String[] split2 = this.strVibrateDataBuffer.toString().split(this.mSeparator);
                this.strVibrateDataBuffer = new StringBuffer();
                for (int i2 = 0; i2 < split2.length; i2++) {
                    if (!split2[i2].toUpperCase().startsWith(upperCase)) {
                        this.strVibrateDataBuffer.append(split2[i2]);
                        this.strVibrateDataBuffer.append(this.mSeparator);
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /* JADX WARN: Removed duplicated region for block: B:26:0x00aa  */
    /* JADX WARN: Removed duplicated region for block: B:40:0x0119  */
    /* JADX WARN: Removed duplicated region for block: B:45:0x0137  */
    /* JADX WARN: Removed duplicated region for block: B:48:0x0138 A[SYNTHETIC] */
    /* JADX WARN: Removed duplicated region for block: B:49:0x0124  */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private void initSpeedLine(Canvas canvas, StringBuffer stringBuffer, Path path, Path path2, int[] iArr) {
        StringBuffer stringBuffer2;
        int i;
        boolean z;
        if (stringBuffer.length() > 0) {
            int ceil = ((int) Math.ceil((this.mScreenTime * 1.0f) / this.intervalTime)) + 1;
            int i2 = ceil * 3;
            stringBuffer2 = i2 <= stringBuffer.length() ? new StringBuffer(stringBuffer.substring(0, i2)) : new StringBuffer(stringBuffer.substring(0));
            if (this.mIsAppend && stringBuffer2.length() / 3 < ceil) {
                String stringBuffer3 = stringBuffer2.toString();
                int length = stringBuffer2.length() / 3;
                int i3 = (ceil / length) - 1;
                int i4 = ceil % length;
                for (int i5 = 0; i5 < i3; i5++) {
                    stringBuffer2.append(stringBuffer3);
                }
                stringBuffer2.append(stringBuffer3.substring(0, i4 * 3));
            }
        } else {
            stringBuffer2 = null;
        }
        float f = 0.0f;
        for (int i6 = 0; stringBuffer2 != null && i6 < stringBuffer2.length() / 3; i6++) {
            float f2 = (this.mDistance * i6) + this.mScaleLeftX;
            int i7 = i6 * 3;
            try {
                i = Integer.valueOf(stringBuffer2.substring(i7, i7 + 2), 16).intValue();
            } catch (Exception unused) {
                i = 0;
            }
            int i8 = this.mScaleLeftX;
            if (f2 >= i8) {
                i8 = this.mScaleRightX;
                if (f2 <= i8) {
                    z = false;
                    if (z) {
                        if (i6 < (stringBuffer2.length() / 3) - 1) {
                            int i9 = (i6 + 1) * 3;
                            int intValue = Integer.valueOf(stringBuffer2.substring(i9, i9 + 2), 16).intValue();
                            int abs = Math.abs(intValue - i);
                            int i10 = this.intervalTime;
                            int i11 = (abs * (0 % i10)) / i10;
                            i = intValue > i ? i + i11 : i - i11;
                        } else if (i6 == (stringBuffer2.length() / 3) - 1) {
                            int i12 = (i6 - 1) * 3;
                            int intValue2 = Integer.valueOf(stringBuffer2.substring(i12, i12 + 2), 16).intValue();
                            int abs2 = Math.abs(intValue2 - i);
                            int i13 = this.mScreenTime;
                            int i14 = this.intervalTime;
                            int i15 = (abs2 * (i13 % i14)) / i14;
                            i = intValue2 > i ? intValue2 - i15 : intValue2 + i15;
                        }
                    }
                    float f3 = this.mYHeight;
                    float f4 = (f3 - ((f3 / 100.0f) * i)) + this.mScaleTopY;
                    if (path.isEmpty()) {
                        path.lineTo(f2, f4);
                        if (this.isShadow) {
                            path2.lineTo(f2, f4);
                        }
                    } else {
                        path.moveTo(f2, f4);
                        if (this.isShadow) {
                            path2.moveTo(f2, f4);
                        }
                    }
                    if (i6 != (stringBuffer2.length() / 3) - 1) {
                        f = f2;
                    }
                }
            }
            f2 = i8;
            z = true;
            if (z) {
            }
            float f32 = this.mYHeight;
            float f42 = (f32 - ((f32 / 100.0f) * i)) + this.mScaleTopY;
            if (path.isEmpty()) {
            }
            if (i6 != (stringBuffer2.length() / 3) - 1) {
            }
        }
        if (this.isShadow) {
            canvas.drawPath(path2, this.mShadowPaint);
        }
        if (iArr != null) {
            this.mPaint.setShader(new LinearGradient(0.0f, 0.0f, f, 0.0f, iArr, new float[]{0.0f, 1.0f}, Shader.TileMode.CLAMP));
        }
        canvas.drawPath(path, this.mPaint);
    }

    private void initType(AttributeSet attributeSet) {
        if (attributeSet != null) {
            TypedArray obtainStyledAttributes = getContext().obtainStyledAttributes(attributeSet, R.styleable.LinePointsView);
            this.isShadow = obtainStyledAttributes.getBoolean(R.styleable.LinePointsView_isShadow, true);
            obtainStyledAttributes.recycle();
        }
    }

    public boolean isStretch() {
        return this.isStretch;
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        drawLine(canvas);
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        this.mWidth = getMeasuredWidth();
        int measuredHeight = getMeasuredHeight();
        this.mHeight = measuredHeight;
        int i3 = this.mWidth;
        int i4 = this.mScaleLeft;
        int i5 = this.mScaleRight;
        int i6 = (i3 - i4) - i5;
        this.mScaleWidth = i6;
        int i7 = this.mScaleTop;
        int i8 = this.mScaleBottom;
        int i9 = (measuredHeight - i7) - i8;
        this.mScaleHeight = i9;
        this.mScaleTopY = i7;
        this.mScaleBottomY = measuredHeight - i8;
        this.mScaleLeftX = i4;
        this.mScaleRightX = i6 - i5;
        float f = i9;
        this.mYHeight = f;
        this.mYItemHeight = f / this.mYItemNum;
        caleScale2();
    }

    public void setData(String str, String str2) {
        this.strDataBuffer = new StringBuffer(str);
        this.strVibrateDataBuffer = new StringBuffer(str2);
        initFunData();
        caleScale2();
        invalidate();
    }

    public void setData(String str, String str2, boolean z) {
        this.mIsAppend = z;
        setData(str, str2);
    }

    public void setData(String[] strArr) {
        if (strArr == null || strArr.length == 0) {
            return;
        }
        this.strDataBuffer = new StringBuffer();
        this.strVibrateDataBuffer = new StringBuffer();
        this.strStartPositionDataBuffer = new StringBuffer();
        this.strSmoothDataBuffer = new StringBuffer();
        if (strArr.length != 5 || TextUtils.isEmpty(strArr[4])) {
            if (!TextUtils.isEmpty(strArr[1])) {
                this.strDataBuffer.append(strArr[1]);
            }
            if (!TextUtils.isEmpty(strArr[2])) {
                this.strVibrateDataBuffer.append(strArr[2]);
            }
        } else {
            if (!TextUtils.isEmpty(strArr[4])) {
                this.strDataBuffer.append(strArr[4]);
            }
            if (!TextUtils.isEmpty(strArr[2])) {
                this.strVibrateDataBuffer.append(strArr[2]);
            }
            if (!TextUtils.isEmpty(strArr[1])) {
                this.strStartPositionDataBuffer.append(strArr[1]);
            }
            if (!TextUtils.isEmpty(strArr[3])) {
                this.strSmoothDataBuffer.append(strArr[3]);
            }
        }
        initFunData();
        caleScale2();
        invalidate();
    }

    public void setData(String[] strArr, boolean z) {
        this.mIsAppend = z;
        setData(strArr);
    }

    public void setStretch(boolean z) {
        this.isStretch = z;
        caleScale2();
        StringBuffer stringBuffer = this.strDataBuffer;
        if (stringBuffer == null || stringBuffer.length() <= 0) {
            return;
        }
        invalidate();
    }
}
