package com.hismith.hismithapp.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.graphics.Shader;
import android.text.TextUtils;
import android.util.AttributeSet;
import android.view.View;
import androidx.exifinterface.media.ExifInterface;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* loaded from: classes3.dex */
public class LineChartPlayView3 extends View {
    private String KEY_MAX_POSITION_MAP;
    private String KEY_MIN_POSITION_MAP;
    private String KEY_SMOOTH_MAP;
    private String KEY_SPEED_MAP;
    private String KEY_VIBRATE_MAP;
    private float height;
    private int intervalTime;
    private float intervalX;
    private float intervalY;
    private boolean isRun;
    private Path linePath;
    private Paint mCirclePaint;
    private Thread mDanThread;
    private int mEndTime;
    private double mLastSmooth;
    private double mLastSpeed;
    private double mLastStartPosition;
    private double mLastStopPosition;
    private double mLastVibrate;
    private Paint mMaxPositionCirclePaint;
    private Paint mMaxPositionPaint;
    private Paint mMaxPositionPlayPaint;
    private List<Float> mMaxPositionPoint;
    private Path mMaxPositionRunLinePath;
    private Paint mMinPositionCirclePaint;
    private Paint mMinPositionPaint;
    private Paint mMinPositionPlayPaint;
    private List<Float> mMinPositionPoint;
    private Path mMinPositionRunLinePath;
    private String mModeData;
    private String[] mModeDataArr;
    private Paint mPaint;
    private Paint mPlayPaint;
    private long mPlayRefreshTime;
    private Thread mPlayThread;
    private List<Float> mPoint;
    private Paint mReversalPaint;
    private Path mReversalPath;
    private Path mRunLinePath;
    private Shader mShader;
    private Paint mSmoothCirclePaint;
    private Paint mSmoothPaint;
    private Paint mSmoothPlayPaint;
    private List<Float> mSmoothPoint;
    private Path mSmoothRunLinePath;
    private Shader mSmoothShader;
    private Shader mStartPositionShader;
    private int mStartTime;
    private Shader mStopPositionShader;
    private Paint mVibrateCirclePaint;
    private Paint mVibratePaint;
    private Paint mVibratePlayPaint;
    private List<Float> mVibratePoint;
    private Path mVibrateRunLinePath;
    private Shader mVibrateShader;
    private Path maxPositionLinePath;
    private Path minPositionLinePath;
    private int offsetY;
    private OnPlayListener onPlayListener;
    private float radius;
    private Path smoothLinePath;
    private int speedIndex;
    private float time;
    private int totalPoint;
    private Path vibrateLinePath;
    private float width;

    public interface OnPlayListener {
        void onComplete();

        void onPlaying(boolean z);

        void onPositionRangeChange(int i, int i2);

        void onSmoothChange(int i);

        void onSpeedChange(float f);

        void onTimeChange(int i);

        void onVibrateChange(float f);
    }

    public LineChartPlayView3(Context context) {
        this(context, null);
    }

    public LineChartPlayView3(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public LineChartPlayView3(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.offsetY = 10;
        this.intervalTime = 5;
        this.totalPoint = 60;
        this.time = 30.0f;
        this.mStartTime = 0;
        this.mEndTime = 0;
        this.speedIndex = 0;
        this.isRun = true;
        this.mPlayRefreshTime = 50L;
        this.mLastSpeed = -1.0d;
        this.mLastVibrate = -1.0d;
        this.mLastStartPosition = -1.0d;
        this.mLastStopPosition = -1.0d;
        this.mLastSmooth = -1.0d;
        this.KEY_SPEED_MAP = "KEY_SPEED_MAP";
        this.KEY_VIBRATE_MAP = "KEY_VIBRATE_MAP";
        this.KEY_MAX_POSITION_MAP = "KEY_MAX_POSITION_MAP";
        this.KEY_MIN_POSITION_MAP = "KEY_MIN_POSITION_MAP";
        this.KEY_SMOOTH_MAP = "KEY_SMOOTH_MAP";
        init();
    }

    private void drawReversal(int i, float f) {
        if ((i & 3840) > 0) {
            this.mReversalPath.addCircle(f, this.height - ((i & 255) * this.intervalY), this.radius, Path.Direction.CW);
        }
    }

    private Rect getTextBounds(String str, Paint paint) {
        Rect rect = new Rect();
        paint.getTextBounds(str, 0, str.length(), rect);
        return rect;
    }

    public static int getTextWidth(Paint paint, String str) {
        float f = 0.0f;
        if (str != null && !str.equals("")) {
            int length = str.length();
            paint.getTextWidths(str, new float[length]);
            for (int i = 0; i < length; i++) {
                f = (float) (f + Math.ceil(r2[i]));
            }
        }
        return (int) f;
    }

    private void init() {
        Paint paint = new Paint();
        this.mPaint = paint;
        paint.setAntiAlias(true);
        this.mPaint.setStrokeWidth(3.0f);
        this.mPaint.setStyle(Paint.Style.STROKE);
        this.mPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint2 = new Paint();
        this.mPlayPaint = paint2;
        paint2.setAntiAlias(true);
        this.mPlayPaint.setStyle(Paint.Style.STROKE);
        this.mPlayPaint.setStrokeWidth(3.0f);
        this.mPlayPaint.setColor(getResources().getColor(R.color.play_mode_line_circle));
        this.mPlayPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mPlayPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint3 = new Paint();
        this.mCirclePaint = paint3;
        paint3.setAntiAlias(true);
        this.mCirclePaint.setStyle(Paint.Style.FILL);
        this.mCirclePaint.setColor(getResources().getColor(R.color.play_mode_line_circle));
        Paint paint4 = new Paint();
        this.mReversalPaint = paint4;
        paint4.setAntiAlias(true);
        this.mReversalPaint.setStyle(Paint.Style.FILL);
        this.mReversalPaint.setColor(Color.parseColor("#FF0000"));
        this.mRunLinePath = new Path();
        this.linePath = new Path();
        this.mReversalPath = new Path();
        Paint paint5 = new Paint();
        this.mVibratePaint = paint5;
        paint5.setAntiAlias(true);
        this.mVibratePaint.setStrokeWidth(3.0f);
        this.mVibratePaint.setStyle(Paint.Style.STROKE);
        this.mVibratePaint.setStrokeJoin(Paint.Join.ROUND);
        this.mVibratePaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint6 = new Paint();
        this.mVibratePlayPaint = paint6;
        paint6.setAntiAlias(true);
        this.mVibratePlayPaint.setStyle(Paint.Style.STROKE);
        this.mVibratePlayPaint.setStrokeWidth(3.0f);
        this.mVibratePlayPaint.setColor(getResources().getColor(R.color.play_mode_vibrate_line_circle));
        this.mVibratePlayPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mVibratePlayPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint7 = new Paint();
        this.mVibrateCirclePaint = paint7;
        paint7.setAntiAlias(true);
        this.mVibrateCirclePaint.setStyle(Paint.Style.FILL);
        this.mVibrateCirclePaint.setColor(getResources().getColor(R.color.play_mode_vibrate_line_circle));
        this.mVibrateRunLinePath = new Path();
        this.vibrateLinePath = new Path();
        Paint paint8 = new Paint();
        this.mMaxPositionPaint = paint8;
        paint8.setAntiAlias(true);
        this.mMaxPositionPaint.setStrokeWidth(3.0f);
        this.mMaxPositionPaint.setStyle(Paint.Style.STROKE);
        this.mMaxPositionPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mMaxPositionPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint9 = new Paint();
        this.mMaxPositionPlayPaint = paint9;
        paint9.setAntiAlias(true);
        this.mMaxPositionPlayPaint.setStyle(Paint.Style.STROKE);
        this.mMaxPositionPlayPaint.setStrokeWidth(3.0f);
        this.mMaxPositionPlayPaint.setColor(getResources().getColor(R.color.play_mode_stop_position_line_circle));
        this.mMaxPositionPlayPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mMaxPositionPlayPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint10 = new Paint();
        this.mMaxPositionCirclePaint = paint10;
        paint10.setAntiAlias(true);
        this.mMaxPositionCirclePaint.setStyle(Paint.Style.FILL);
        this.mMaxPositionCirclePaint.setColor(getResources().getColor(R.color.play_mode_stop_position_line_circle));
        this.mMaxPositionRunLinePath = new Path();
        this.maxPositionLinePath = new Path();
        Paint paint11 = new Paint();
        this.mMinPositionPaint = paint11;
        paint11.setAntiAlias(true);
        this.mMinPositionPaint.setStrokeWidth(3.0f);
        this.mMinPositionPaint.setStyle(Paint.Style.STROKE);
        this.mMinPositionPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mMinPositionPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint12 = new Paint();
        this.mMinPositionPlayPaint = paint12;
        paint12.setAntiAlias(true);
        this.mMinPositionPlayPaint.setStyle(Paint.Style.STROKE);
        this.mMinPositionPlayPaint.setStrokeWidth(3.0f);
        this.mMinPositionPlayPaint.setColor(getResources().getColor(R.color.play_mode_start_position_line_circle));
        this.mMinPositionPlayPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mMinPositionPlayPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint13 = new Paint();
        this.mMinPositionCirclePaint = paint13;
        paint13.setAntiAlias(true);
        this.mMinPositionCirclePaint.setStyle(Paint.Style.FILL);
        this.mMinPositionCirclePaint.setColor(getResources().getColor(R.color.play_mode_start_position_line_circle));
        this.mMinPositionRunLinePath = new Path();
        this.minPositionLinePath = new Path();
        Paint paint14 = new Paint();
        this.mSmoothPaint = paint14;
        paint14.setAntiAlias(true);
        this.mSmoothPaint.setStrokeWidth(3.0f);
        this.mSmoothPaint.setStyle(Paint.Style.STROKE);
        this.mSmoothPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mSmoothPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint15 = new Paint();
        this.mSmoothPlayPaint = paint15;
        paint15.setAntiAlias(true);
        this.mSmoothPlayPaint.setStyle(Paint.Style.STROKE);
        this.mSmoothPlayPaint.setStrokeWidth(3.0f);
        this.mSmoothPlayPaint.setColor(getResources().getColor(R.color.play_mode_smooth_line_circle));
        this.mSmoothPlayPaint.setStrokeJoin(Paint.Join.ROUND);
        this.mSmoothPlayPaint.setStrokeCap(Paint.Cap.ROUND);
        Paint paint16 = new Paint();
        this.mSmoothCirclePaint = paint16;
        paint16.setAntiAlias(true);
        this.mSmoothCirclePaint.setStyle(Paint.Style.FILL);
        this.mSmoothCirclePaint.setColor(getResources().getColor(R.color.play_mode_smooth_line_circle));
        this.mSmoothRunLinePath = new Path();
        this.smoothLinePath = new Path();
        this.mPoint = new ArrayList();
        this.mVibratePoint = new ArrayList();
        this.mMaxPositionPoint = new ArrayList();
        this.mMinPositionPoint = new ArrayList();
        this.mSmoothPoint = new ArrayList();
    }

    /* JADX WARN: Removed duplicated region for block: B:102:0x0991  */
    /* JADX WARN: Removed duplicated region for block: B:99:0x0987 A[Catch: Exception -> 0x099f, TRY_LEAVE, TryCatch #9 {Exception -> 0x099f, blocks: (B:94:0x0976, B:96:0x097c, B:99:0x0987), top: B:93:0x0976 }] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private Map<String, List<Float>> initPoint(int i) {
        HashMap hashMap;
        ArrayList arrayList;
        ArrayList arrayList2;
        ArrayList arrayList3;
        ArrayList arrayList4;
        int i2;
        ArrayList arrayList5;
        String substring;
        int i3;
        ArrayList arrayList6;
        ArrayList arrayList7;
        ArrayList arrayList8;
        int i4;
        ArrayList arrayList9;
        ArrayList arrayList10;
        ArrayList arrayList11;
        Float f;
        int i5;
        int i6;
        int i7;
        int i8;
        HashMap hashMap2 = new HashMap();
        ArrayList arrayList12 = new ArrayList();
        ArrayList arrayList13 = new ArrayList();
        ArrayList arrayList14 = new ArrayList();
        ArrayList arrayList15 = new ArrayList();
        ArrayList arrayList16 = new ArrayList();
        ArrayList arrayList17 = new ArrayList();
        ArrayList arrayList18 = new ArrayList();
        ArrayList arrayList19 = new ArrayList();
        ArrayList arrayList20 = new ArrayList();
        ArrayList arrayList21 = new ArrayList();
        int i9 = this.intervalTime;
        int i10 = i;
        int i11 = 0;
        while (true) {
            String[] strArr = this.mModeDataArr;
            int i12 = i10;
            if (i11 >= strArr.length) {
                hashMap = hashMap2;
                arrayList = arrayList15;
                arrayList2 = arrayList14;
                arrayList3 = arrayList16;
                break;
            }
            if (!TextUtils.isEmpty(strArr[i11]) && this.mModeDataArr[i11].length() > 3) {
                i2 = i11;
                try {
                    substring = this.mModeDataArr[i11].substring(0, 1);
                    hashMap = hashMap2;
                } catch (Exception unused) {
                    hashMap = hashMap2;
                }
                try {
                    arrayList6 = arrayList16;
                } catch (Exception unused2) {
                    arrayList = arrayList15;
                    arrayList5 = arrayList18;
                    arrayList4 = arrayList19;
                    arrayList2 = arrayList14;
                    arrayList3 = arrayList16;
                    i10 = i12;
                    arrayList18 = arrayList5;
                    arrayList19 = arrayList4;
                    i11 = i2 + 1;
                    arrayList16 = arrayList3;
                    arrayList14 = arrayList2;
                    arrayList15 = arrayList;
                    hashMap2 = hashMap;
                }
                if (Integer.valueOf(this.mModeDataArr[i2].substring(1, 3), 16).intValue() == 255) {
                    try {
                    } catch (Exception unused3) {
                        i10 = i12;
                        arrayList = arrayList15;
                        arrayList5 = arrayList18;
                        arrayList4 = arrayList19;
                        arrayList2 = arrayList14;
                        arrayList3 = arrayList6;
                    }
                    if ("1".equals(substring)) {
                        int intValue = Integer.valueOf(this.mModeDataArr[i2].substring(3, 5), 16).intValue();
                        ArrayList arrayList22 = new ArrayList();
                        if (arrayList17.size() > 0) {
                            int size = arrayList17.size() - 1;
                            for (i3 = 1; size >= 0 && ((List) arrayList17.get(size)).size() > i3; i3 = 1) {
                                int intValue2 = ((Float) ((List) arrayList17.get(size)).get(((List) arrayList17.get(size)).size() - 1)).intValue();
                                if (intValue == 0) {
                                    intValue2 |= 256;
                                    arrayList22.add(Float.valueOf(256.0f));
                                } else if (intValue == 1) {
                                    intValue2 |= 512;
                                    arrayList22.add(Float.valueOf(512.0f));
                                }
                                if (i12 <= 0 && arrayList12.size() > 0) {
                                    arrayList12.set(arrayList12.size() - 1, Float.valueOf(intValue2 * 1.0f));
                                }
                                size--;
                            }
                        } else if (intValue == 0) {
                            arrayList22.add(Float.valueOf(256.0f));
                            if (i12 <= 0) {
                                arrayList12.add(Float.valueOf(256.0f));
                            }
                        } else if (intValue == 1) {
                            arrayList22.add(Float.valueOf(512.0f));
                            if (i12 <= 0) {
                                arrayList12.add(Float.valueOf(512.0f));
                            }
                        }
                        arrayList17.add(arrayList22);
                    } else if ("2".equals(substring)) {
                        int intValue3 = Integer.valueOf(this.mModeDataArr[i2].substring(3, 5), 16).intValue();
                        int intValue4 = Integer.valueOf(this.mModeDataArr[i2].substring(7, 9), 16).intValue();
                        ArrayList arrayList23 = new ArrayList();
                        ArrayList arrayList24 = new ArrayList();
                        if (arrayList17.size() > 0) {
                            int size2 = arrayList17.size() - 1;
                            while (true) {
                                if (size2 < 0) {
                                    arrayList7 = arrayList14;
                                    arrayList8 = arrayList15;
                                    break;
                                }
                                arrayList8 = arrayList15;
                                try {
                                    arrayList7 = arrayList14;
                                    if (((List) arrayList17.get(size2)).size() <= 1) {
                                        break;
                                    }
                                    try {
                                        int intValue5 = ((Float) ((List) arrayList17.get(size2)).get(((List) arrayList17.get(size2)).size() - 1)).intValue();
                                        int i13 = size2;
                                        int intValue6 = ((Float) ((List) arrayList18.get(size2)).get(((List) arrayList18.get(size2)).size() - 1)).intValue();
                                        if (intValue3 == 0) {
                                            intValue5 |= 256;
                                            arrayList23.add(Float.valueOf(256.0f));
                                        } else if (intValue3 == 1) {
                                            intValue5 |= 512;
                                            arrayList23.add(Float.valueOf(512.0f));
                                        }
                                        if (intValue4 == 0) {
                                            intValue6 |= 256;
                                            arrayList24.add(Float.valueOf(256.0f));
                                        } else if (intValue4 == 1) {
                                            intValue6 |= 512;
                                            arrayList24.add(Float.valueOf(512.0f));
                                        }
                                        if (i12 <= 0) {
                                            if (arrayList12.size() > 0) {
                                                arrayList12.set(arrayList12.size() - 1, Float.valueOf(intValue5 * 1.0f));
                                            }
                                            if (arrayList13.size() > 0) {
                                                arrayList13.set(arrayList13.size() - 1, Float.valueOf(intValue6 * 1.0f));
                                            }
                                        }
                                        size2 = i13 - 1;
                                        arrayList15 = arrayList8;
                                        arrayList14 = arrayList7;
                                    } catch (Exception unused4) {
                                    }
                                } catch (Exception unused5) {
                                    i10 = i12;
                                    arrayList2 = arrayList14;
                                    arrayList5 = arrayList18;
                                    arrayList4 = arrayList19;
                                    arrayList3 = arrayList6;
                                    arrayList = arrayList8;
                                }
                            }
                        } else {
                            arrayList7 = arrayList14;
                            arrayList8 = arrayList15;
                            if (intValue3 == 0) {
                                arrayList23.add(Float.valueOf(256.0f));
                                if (i12 <= 0) {
                                    arrayList12.add(Float.valueOf(256.0f));
                                }
                            } else if (intValue3 == 1) {
                                arrayList23.add(Float.valueOf(512.0f));
                                if (i12 <= 0) {
                                    arrayList12.add(Float.valueOf(512.0f));
                                }
                            }
                            if (intValue4 == 0) {
                                arrayList24.add(Float.valueOf(256.0f));
                                if (i12 <= 0) {
                                    arrayList13.add(Float.valueOf(256.0f));
                                }
                            } else if (intValue4 == 1) {
                                arrayList24.add(Float.valueOf(512.0f));
                                if (i12 <= 0) {
                                    arrayList13.add(Float.valueOf(512.0f));
                                }
                            }
                        }
                        arrayList17.add(arrayList23);
                        arrayList18.add(arrayList24);
                        i10 = i12;
                        arrayList5 = arrayList18;
                        arrayList4 = arrayList19;
                        arrayList3 = arrayList6;
                        arrayList = arrayList8;
                        arrayList2 = arrayList7;
                        arrayList18 = arrayList5;
                        arrayList19 = arrayList4;
                        i11 = i2 + 1;
                        arrayList16 = arrayList3;
                        arrayList14 = arrayList2;
                        arrayList15 = arrayList;
                        hashMap2 = hashMap;
                    }
                    arrayList7 = arrayList14;
                    arrayList8 = arrayList15;
                    i10 = i12;
                    arrayList5 = arrayList18;
                    arrayList4 = arrayList19;
                    arrayList3 = arrayList6;
                    arrayList = arrayList8;
                    arrayList2 = arrayList7;
                    arrayList18 = arrayList5;
                    arrayList19 = arrayList4;
                    i11 = i2 + 1;
                    arrayList16 = arrayList3;
                    arrayList14 = arrayList2;
                    arrayList15 = arrayList;
                    hashMap2 = hashMap;
                } else {
                    arrayList7 = arrayList14;
                    arrayList8 = arrayList15;
                    try {
                        if ("2".equals(substring)) {
                            float intValue7 = Integer.valueOf(this.mModeDataArr[i2].substring(1, 3), 16).intValue();
                            float intValue8 = Integer.valueOf(this.mModeDataArr[i2].substring(3, 5), 16).intValue();
                            float intValue9 = Integer.valueOf(this.mModeDataArr[i2].substring(5, 7), 16).intValue();
                            float intValue10 = Integer.valueOf(this.mModeDataArr[i2].substring(7, 9), 16).intValue();
                            float intValue11 = (Integer.valueOf(this.mModeDataArr[i2].substring(9, 11), 16).intValue() * 100) + (Integer.valueOf(this.mModeDataArr[i2].substring(11, 12), 16).intValue() * 10.0f);
                            ArrayList arrayList25 = new ArrayList();
                            ArrayList arrayList26 = new ArrayList();
                            if (intValue11 > i9) {
                                float ceil = (float) Math.ceil(intValue11 / r15);
                                float f2 = (intValue8 - intValue7) / ceil;
                                float f3 = (intValue10 - intValue9) / ceil;
                                int i14 = 0;
                                while (true) {
                                    if (i14 >= ceil) {
                                        break;
                                    }
                                    float f4 = f2;
                                    float ceil2 = (float) Math.ceil(intValue7 + (r4 * f2));
                                    if (ceil2 > 100.0f) {
                                        ceil2 = 100.0f;
                                    }
                                    arrayList25.add(Float.valueOf(ceil2));
                                    float ceil3 = (float) Math.ceil((r4 * f3) + intValue9);
                                    if (ceil3 > 100.0f) {
                                        ceil3 = 100.0f;
                                    }
                                    arrayList26.add(Float.valueOf(ceil3));
                                    i14++;
                                    f2 = f4;
                                }
                            } else {
                                if (intValue7 == intValue8) {
                                    arrayList25.add(Float.valueOf(intValue7));
                                } else {
                                    arrayList25.add(Float.valueOf(intValue7));
                                    arrayList25.add(Float.valueOf(intValue8));
                                }
                                if (intValue9 == intValue10) {
                                    arrayList26.add(Float.valueOf(intValue9));
                                } else {
                                    arrayList26.add(Float.valueOf(intValue9));
                                    arrayList26.add(Float.valueOf(intValue10));
                                }
                            }
                            arrayList17.add(arrayList25);
                            arrayList18.add(arrayList26);
                            int i15 = i12;
                            for (int i16 = 0; i16 < arrayList25.size() && arrayList12.size() < this.totalPoint; i16++) {
                                try {
                                    if (i15 <= 0) {
                                        arrayList12.add((Float) arrayList25.get(i16));
                                        arrayList13.add((Float) arrayList26.get(i16));
                                    } else {
                                        i15 -= i9;
                                    }
                                } catch (Exception unused6) {
                                    i10 = i15;
                                }
                            }
                            i10 = i15;
                            arrayList5 = arrayList18;
                            arrayList4 = arrayList19;
                            arrayList3 = arrayList6;
                            arrayList = arrayList8;
                            arrayList2 = arrayList7;
                        } else {
                            if (!"1".equals(substring)) {
                                try {
                                    if (!ExifInterface.GPS_MEASUREMENT_3D.equals(substring)) {
                                        if ("S".equals(substring)) {
                                            try {
                                                float intValue12 = Integer.valueOf(this.mModeDataArr[i2].substring(1, 3), 16).intValue();
                                                float intValue13 = Integer.valueOf(this.mModeDataArr[i2].substring(3, 5), 16).intValue();
                                                float intValue14 = Integer.valueOf(this.mModeDataArr[i2].substring(5, 7), 16).intValue();
                                                float intValue15 = Integer.valueOf(this.mModeDataArr[i2].substring(7, 9), 16).intValue();
                                                float intValue16 = Integer.valueOf(this.mModeDataArr[i2].substring(9, 10), 16).intValue();
                                                float intValue17 = (Integer.valueOf(this.mModeDataArr[i2].substring(10, 12), 16).intValue() * 100) + (Integer.valueOf(this.mModeDataArr[i2].substring(12, 13), 16).intValue() * 10.0f);
                                                ArrayList arrayList27 = new ArrayList();
                                                ArrayList arrayList28 = new ArrayList();
                                                ArrayList arrayList29 = new ArrayList();
                                                ArrayList arrayList30 = new ArrayList();
                                                if (intValue17 > i9) {
                                                    float ceil4 = (float) Math.ceil(intValue17 / r5);
                                                    float f5 = (intValue15 - intValue14) / ceil4;
                                                    int i17 = 0;
                                                    while (true) {
                                                        if (i17 >= ceil4) {
                                                            break;
                                                        }
                                                        int i18 = i17;
                                                        float ceil5 = (float) Math.ceil(intValue14 + (r6 * f5));
                                                        if (ceil5 > 100.0f) {
                                                            ceil5 = 100.0f;
                                                        }
                                                        arrayList27.add(Float.valueOf(ceil5));
                                                        arrayList28.add(Float.valueOf(intValue13));
                                                        arrayList29.add(Float.valueOf(intValue12));
                                                        arrayList30.add(Float.valueOf(intValue16));
                                                        i17 = i18 + 1;
                                                    }
                                                } else {
                                                    arrayList27.add(Float.valueOf(intValue14));
                                                    arrayList28.add(Float.valueOf(intValue13));
                                                    arrayList29.add(Float.valueOf(intValue12));
                                                    arrayList30.add(Float.valueOf(intValue16));
                                                }
                                                arrayList17.add(arrayList27);
                                                arrayList19.add(arrayList28);
                                                arrayList20.add(arrayList29);
                                                arrayList21.add(arrayList30);
                                                int i19 = i12;
                                                int i20 = 0;
                                                while (true) {
                                                    try {
                                                        if (i20 >= arrayList27.size()) {
                                                            i4 = i19;
                                                            break;
                                                        }
                                                        i4 = i19;
                                                        if (arrayList12.size() >= this.totalPoint) {
                                                            break;
                                                        }
                                                        if (i4 <= 0) {
                                                            arrayList12.add((Float) arrayList27.get(i20));
                                                            arrayList11 = arrayList7;
                                                            try {
                                                                arrayList11.add((Float) arrayList28.get(i20));
                                                                Float f6 = (Float) arrayList29.get(i20);
                                                                arrayList9 = arrayList29;
                                                                arrayList = arrayList8;
                                                                try {
                                                                    arrayList.add(f6);
                                                                    f = (Float) arrayList30.get(i20);
                                                                    arrayList10 = arrayList30;
                                                                    arrayList3 = arrayList6;
                                                                } catch (Exception unused7) {
                                                                    arrayList3 = arrayList6;
                                                                }
                                                                try {
                                                                    arrayList3.add(f);
                                                                    i19 = i4;
                                                                } catch (Exception unused8) {
                                                                    i10 = i4;
                                                                    arrayList5 = arrayList18;
                                                                    arrayList4 = arrayList19;
                                                                    arrayList2 = arrayList11;
                                                                    arrayList18 = arrayList5;
                                                                    arrayList19 = arrayList4;
                                                                    i11 = i2 + 1;
                                                                    arrayList16 = arrayList3;
                                                                    arrayList14 = arrayList2;
                                                                    arrayList15 = arrayList;
                                                                    hashMap2 = hashMap;
                                                                }
                                                            } catch (Exception unused9) {
                                                                arrayList3 = arrayList6;
                                                                arrayList = arrayList8;
                                                            }
                                                        } else {
                                                            arrayList9 = arrayList29;
                                                            arrayList10 = arrayList30;
                                                            arrayList3 = arrayList6;
                                                            arrayList = arrayList8;
                                                            arrayList11 = arrayList7;
                                                            i19 = i4 - i9;
                                                        }
                                                        i20++;
                                                        arrayList8 = arrayList;
                                                        arrayList6 = arrayList3;
                                                        arrayList7 = arrayList11;
                                                        arrayList29 = arrayList9;
                                                        arrayList30 = arrayList10;
                                                    } catch (Exception unused10) {
                                                        i12 = i19;
                                                        arrayList3 = arrayList6;
                                                        arrayList = arrayList8;
                                                        i10 = i12;
                                                        arrayList5 = arrayList18;
                                                        arrayList4 = arrayList19;
                                                        arrayList2 = arrayList7;
                                                        arrayList18 = arrayList5;
                                                        arrayList19 = arrayList4;
                                                        i11 = i2 + 1;
                                                        arrayList16 = arrayList3;
                                                        arrayList14 = arrayList2;
                                                        arrayList15 = arrayList;
                                                        hashMap2 = hashMap;
                                                    }
                                                }
                                                arrayList3 = arrayList6;
                                                arrayList = arrayList8;
                                                i10 = i4;
                                                arrayList5 = arrayList18;
                                                arrayList4 = arrayList19;
                                                arrayList2 = arrayList7;
                                            } catch (Exception unused11) {
                                            }
                                        } else {
                                            arrayList3 = arrayList6;
                                            arrayList = arrayList8;
                                            arrayList2 = arrayList7;
                                            try {
                                                if ("0".equals(substring)) {
                                                    int intValue18 = Integer.valueOf(this.mModeDataArr[i2].substring(1, 3), 16).intValue();
                                                    int intValue19 = Integer.valueOf(this.mModeDataArr[i2].substring(3, 5), 16).intValue();
                                                    int intValue20 = Integer.valueOf(this.mModeDataArr[i2].substring(5, 7), 16).intValue();
                                                    i10 = i12;
                                                    int i21 = 0;
                                                    while (true) {
                                                        if (i21 >= intValue20) {
                                                            i12 = i10;
                                                            break;
                                                        }
                                                        int i22 = intValue20;
                                                        try {
                                                            i12 = i10;
                                                            if (arrayList12.size() >= this.totalPoint) {
                                                                break;
                                                            }
                                                            int i23 = intValue18;
                                                            while (true) {
                                                                i10 = i12;
                                                                if (i23 > intValue19) {
                                                                    i5 = intValue19;
                                                                    break;
                                                                }
                                                                int i24 = i23;
                                                                try {
                                                                    i5 = intValue19;
                                                                    if (arrayList12.size() >= this.totalPoint) {
                                                                        break;
                                                                    }
                                                                    int i25 = 0;
                                                                    while (true) {
                                                                        int i26 = i24 - 1;
                                                                        i12 = i10;
                                                                        if (i25 >= ((List) arrayList17.get(i26)).size()) {
                                                                            i6 = i21;
                                                                            break;
                                                                        }
                                                                        i6 = i21;
                                                                        if (arrayList12.size() >= this.totalPoint) {
                                                                            break;
                                                                        }
                                                                        if (i12 <= 0) {
                                                                            if (((Float) ((List) arrayList17.get(i26)).get(i25)).floatValue() <= 100.0f) {
                                                                                arrayList12.add((Float) ((List) arrayList17.get(i26)).get(i25));
                                                                            } else if (arrayList12.size() > 0) {
                                                                                arrayList12.set(arrayList12.size() - 1, Float.valueOf(((Float) arrayList12.get(arrayList12.size() - 1)).intValue() | ((Float) ((List) arrayList17.get(i26)).get(i25)).intValue()));
                                                                            }
                                                                            if (notAnArray(arrayList18, i26)) {
                                                                                if (((Float) ((List) arrayList18.get(i26)).get(i25)).floatValue() <= 100.0f) {
                                                                                    arrayList13.add((Float) ((List) arrayList18.get(i26)).get(i25));
                                                                                } else if (arrayList13.size() > 0) {
                                                                                    arrayList13.set(arrayList13.size() - 1, Float.valueOf(((Float) arrayList13.get(arrayList13.size() - 1)).intValue() | ((Float) ((List) arrayList18.get(i26)).get(i25)).intValue()));
                                                                                }
                                                                            }
                                                                            if (notAnArray(arrayList19, i26)) {
                                                                                if (((Float) ((List) arrayList19.get(i26)).get(i25)).floatValue() <= 100.0f) {
                                                                                    arrayList2.add((Float) ((List) arrayList19.get(i26)).get(i25));
                                                                                } else if (arrayList2.size() > 0) {
                                                                                    arrayList2.set(arrayList2.size() - 1, Float.valueOf(((Float) arrayList2.get(arrayList2.size() - 1)).intValue() | ((Float) ((List) arrayList19.get(i26)).get(i25)).intValue()));
                                                                                }
                                                                            }
                                                                            if (notAnArray(arrayList20, i26)) {
                                                                                if (((Float) ((List) arrayList20.get(i26)).get(i25)).floatValue() <= 100.0f) {
                                                                                    arrayList.add((Float) ((List) arrayList20.get(i26)).get(i25));
                                                                                } else if (arrayList.size() > 0) {
                                                                                    arrayList.set(arrayList.size() - 1, Float.valueOf(((Float) arrayList.get(arrayList.size() - 1)).intValue() | ((Float) ((List) arrayList20.get(i26)).get(i25)).intValue()));
                                                                                }
                                                                            }
                                                                            if (notAnArray(arrayList21, i26)) {
                                                                                if (((Float) ((List) arrayList21.get(i26)).get(i25)).floatValue() <= 100.0f) {
                                                                                    arrayList3.add((Float) ((List) arrayList21.get(i26)).get(i25));
                                                                                } else if (arrayList3.size() > 0) {
                                                                                    arrayList3.set(arrayList3.size() - 1, Float.valueOf(((Float) ((List) arrayList21.get(i26)).get(i25)).intValue() | ((Float) arrayList3.get(arrayList3.size() - 1)).intValue()));
                                                                                }
                                                                            }
                                                                            i10 = i12;
                                                                        } else {
                                                                            i10 = i12 - i9;
                                                                        }
                                                                        i25++;
                                                                        i21 = i6;
                                                                    }
                                                                    i23 = i24 + 1;
                                                                    intValue19 = i5;
                                                                    i21 = i6;
                                                                } catch (Exception unused12) {
                                                                    arrayList5 = arrayList18;
                                                                    arrayList4 = arrayList19;
                                                                    arrayList18 = arrayList5;
                                                                    arrayList19 = arrayList4;
                                                                    i11 = i2 + 1;
                                                                    arrayList16 = arrayList3;
                                                                    arrayList14 = arrayList2;
                                                                    arrayList15 = arrayList;
                                                                    hashMap2 = hashMap;
                                                                }
                                                            }
                                                            i21++;
                                                            intValue20 = i22;
                                                            intValue19 = i5;
                                                        } catch (Exception unused13) {
                                                        }
                                                    }
                                                }
                                                i10 = i12;
                                                arrayList5 = arrayList18;
                                                arrayList4 = arrayList19;
                                            } catch (Exception unused14) {
                                                i10 = i12;
                                                arrayList5 = arrayList18;
                                                arrayList4 = arrayList19;
                                                arrayList18 = arrayList5;
                                                arrayList19 = arrayList4;
                                                i11 = i2 + 1;
                                                arrayList16 = arrayList3;
                                                arrayList14 = arrayList2;
                                                arrayList15 = arrayList;
                                                hashMap2 = hashMap;
                                            }
                                        }
                                    }
                                } catch (Exception unused15) {
                                    arrayList3 = arrayList6;
                                    arrayList = arrayList8;
                                    arrayList2 = arrayList7;
                                }
                            }
                            arrayList3 = arrayList6;
                            arrayList = arrayList8;
                            arrayList2 = arrayList7;
                            try {
                                float intValue21 = Integer.valueOf(this.mModeDataArr[i2].substring(1, 3), 16).intValue();
                                float intValue22 = Integer.valueOf(this.mModeDataArr[i2].substring(3, 5), 16).intValue();
                                float intValue23 = (Integer.valueOf(this.mModeDataArr[i2].substring(5, 7), 16).intValue() * 100) + (Integer.valueOf(this.mModeDataArr[i2].substring(7, 8), 16).intValue() * 10.0f);
                                ArrayList arrayList31 = new ArrayList();
                                float f7 = i9;
                                try {
                                    if (intValue23 <= f7) {
                                        if (intValue21 == intValue22) {
                                            arrayList31.add(Float.valueOf(intValue21));
                                        } else {
                                            arrayList31.add(Float.valueOf(intValue21));
                                            arrayList31.add(Float.valueOf(intValue22));
                                        }
                                        arrayList5 = arrayList18;
                                        arrayList4 = arrayList19;
                                    } else {
                                        float f8 = intValue23 / f7;
                                        arrayList5 = arrayList18;
                                        arrayList4 = arrayList19;
                                        float ceil6 = (float) Math.ceil(f8);
                                        float f9 = (intValue22 - intValue21) / ceil6;
                                        int i27 = 0;
                                        while (true) {
                                            if (i27 < ceil6) {
                                                int i28 = i27;
                                                float ceil7 = (float) Math.ceil(intValue21 + (r9 * f9));
                                                if (ceil7 > 100.0f) {
                                                    ceil7 = 100.0f;
                                                }
                                                arrayList31.add(Float.valueOf(ceil7));
                                                i27 = i28 + 1;
                                            }
                                        }
                                        arrayList17.add(arrayList31);
                                        i7 = i12;
                                        for (i8 = 0; i8 < arrayList31.size() && arrayList12.size() < this.totalPoint; i8++) {
                                            try {
                                                if (i7 > 0) {
                                                    arrayList12.add((Float) arrayList31.get(i8));
                                                } else {
                                                    i7 -= i9;
                                                }
                                            } catch (Exception unused16) {
                                                i10 = i7;
                                            }
                                        }
                                        i10 = i7;
                                    }
                                    arrayList17.add(arrayList31);
                                    i7 = i12;
                                    while (i8 < arrayList31.size()) {
                                        if (i7 > 0) {
                                        }
                                    }
                                    i10 = i7;
                                } catch (Exception unused17) {
                                }
                            } catch (Exception unused18) {
                                arrayList5 = arrayList18;
                                arrayList4 = arrayList19;
                            }
                        }
                        try {
                        } catch (Exception unused19) {
                            continue;
                        }
                    } catch (Exception unused20) {
                        arrayList5 = arrayList18;
                        arrayList4 = arrayList19;
                        arrayList3 = arrayList6;
                        arrayList = arrayList8;
                        arrayList2 = arrayList7;
                    }
                    if (arrayList12.size() >= this.totalPoint) {
                        break;
                    }
                    arrayList18 = arrayList5;
                    arrayList19 = arrayList4;
                    i11 = i2 + 1;
                    arrayList16 = arrayList3;
                    arrayList14 = arrayList2;
                    arrayList15 = arrayList;
                    hashMap2 = hashMap;
                }
            }
            hashMap = hashMap2;
            arrayList = arrayList15;
            arrayList4 = arrayList19;
            i2 = i11;
            arrayList2 = arrayList14;
            arrayList3 = arrayList16;
            arrayList5 = arrayList18;
            i10 = i12;
            arrayList18 = arrayList5;
            arrayList19 = arrayList4;
            i11 = i2 + 1;
            arrayList16 = arrayList3;
            arrayList14 = arrayList2;
            arrayList15 = arrayList;
            hashMap2 = hashMap;
        }
        HashMap hashMap3 = hashMap;
        hashMap3.put(this.KEY_SPEED_MAP, arrayList12);
        hashMap3.put(this.KEY_VIBRATE_MAP, arrayList13);
        hashMap3.put(this.KEY_MAX_POSITION_MAP, arrayList2);
        hashMap3.put(this.KEY_MIN_POSITION_MAP, arrayList);
        hashMap3.put(this.KEY_SMOOTH_MAP, arrayList3);
        return hashMap3;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public boolean initPoint() {
        List<Float> list;
        int i = this.mStartTime;
        int i2 = this.totalPoint;
        int i3 = this.intervalTime;
        boolean z = false;
        if (i <= (i2 / 2) * i3) {
            Map<String, List<Float>> initPoint = initPoint(0);
            this.mPoint = initPoint.get(this.KEY_SPEED_MAP);
            this.mVibratePoint = initPoint.get(this.KEY_VIBRATE_MAP);
            this.mMaxPositionPoint = initPoint.get(this.KEY_MAX_POSITION_MAP);
            this.mMinPositionPoint = initPoint.get(this.KEY_MIN_POSITION_MAP);
            this.mSmoothPoint = initPoint.get(this.KEY_SMOOTH_MAP);
            this.speedIndex = this.mStartTime / this.intervalTime;
        } else {
            int i4 = this.mEndTime;
            if (i4 - i > (i2 / 2) * i3) {
                Map<String, List<Float>> initPoint2 = initPoint(i - ((i2 / 2) * i3));
                this.mPoint = initPoint2.get(this.KEY_SPEED_MAP);
                this.mVibratePoint = initPoint2.get(this.KEY_VIBRATE_MAP);
                this.mMaxPositionPoint = initPoint2.get(this.KEY_MAX_POSITION_MAP);
                this.mMinPositionPoint = initPoint2.get(this.KEY_MIN_POSITION_MAP);
                this.mSmoothPoint = initPoint2.get(this.KEY_SMOOTH_MAP);
                this.speedIndex = this.totalPoint / 2;
            } else {
                int i5 = i4 - (i2 * i3);
                if (i5 < 0) {
                    i5 = 0;
                }
                Map<String, List<Float>> initPoint3 = initPoint(i5);
                this.mPoint = initPoint3.get(this.KEY_SPEED_MAP);
                this.mVibratePoint = initPoint3.get(this.KEY_VIBRATE_MAP);
                this.mMaxPositionPoint = initPoint3.get(this.KEY_MAX_POSITION_MAP);
                this.mMinPositionPoint = initPoint3.get(this.KEY_MIN_POSITION_MAP);
                this.mSmoothPoint = initPoint3.get(this.KEY_SMOOTH_MAP);
                this.speedIndex = this.mPoint.size() - ((this.mEndTime - this.mStartTime) / this.intervalTime);
            }
        }
        LogUtil.e("mPoint:" + Arrays.toString(this.mPoint.toArray()) + ", time:" + this.mStartTime);
        if (this.speedIndex < 0) {
            return true;
        }
        OnPlayListener onPlayListener = this.onPlayListener;
        if (onPlayListener != null) {
            onPlayListener.onTimeChange(this.mStartTime);
        }
        if (this.mPoint.size() <= this.speedIndex) {
            this.mLastSpeed = 0.0d;
            this.mLastVibrate = 0.0d;
            this.mLastStartPosition = 0.0d;
            this.mLastStopPosition = 0.0d;
            this.mLastSmooth = 0.0d;
            this.mStartTime = 0;
            stopSync();
            OnPlayListener onPlayListener2 = this.onPlayListener;
            if (onPlayListener2 != null) {
                onPlayListener2.onComplete();
            }
        } else {
            if (this.mPoint.size() > 0) {
                LogUtil.e("当前速度值:" + this.mPoint.get(this.speedIndex));
                if (this.mLastSpeed != this.mPoint.get(this.speedIndex).floatValue()) {
                    OnPlayListener onPlayListener3 = this.onPlayListener;
                    if (onPlayListener3 != null) {
                        onPlayListener3.onSpeedChange(this.mPoint.get(this.speedIndex).intValue() & 255);
                    }
                    this.mLastSpeed = this.mPoint.get(this.speedIndex).floatValue();
                }
                List<Float> list2 = this.mVibratePoint;
                if (list2 != null) {
                    if (list2.size() > this.speedIndex && this.mLastVibrate != this.mVibratePoint.get(r2).floatValue()) {
                        OnPlayListener onPlayListener4 = this.onPlayListener;
                        if (onPlayListener4 != null) {
                            onPlayListener4.onVibrateChange(this.mVibratePoint.get(this.speedIndex).intValue() & 255);
                        }
                        this.mLastVibrate = this.mVibratePoint.get(this.speedIndex).floatValue();
                    }
                }
                List<Float> list3 = this.mMaxPositionPoint;
                if (list3 != null && list3.size() > this.speedIndex && (list = this.mMinPositionPoint) != null) {
                    int size = list.size();
                    int i6 = this.speedIndex;
                    if (size > i6) {
                        int intValue = this.mMaxPositionPoint.get(i6).intValue() & 255;
                        int intValue2 = this.mMinPositionPoint.get(this.speedIndex).intValue() & 255;
                        if (this.mLastStartPosition != intValue2 || this.mLastStopPosition != intValue) {
                            OnPlayListener onPlayListener5 = this.onPlayListener;
                            if (onPlayListener5 != null) {
                                onPlayListener5.onPositionRangeChange(intValue2, intValue);
                            }
                            this.mLastStartPosition = this.mMinPositionPoint.get(this.speedIndex).floatValue();
                            this.mLastStopPosition = this.mMaxPositionPoint.get(this.speedIndex).floatValue();
                        }
                    }
                }
                List<Float> list4 = this.mSmoothPoint;
                if (list4 != null) {
                    if (list4.size() > this.speedIndex && this.mLastSmooth != this.mSmoothPoint.get(r2).floatValue()) {
                        OnPlayListener onPlayListener6 = this.onPlayListener;
                        if (onPlayListener6 != null) {
                            onPlayListener6.onSmoothChange(this.mSmoothPoint.get(this.speedIndex).intValue() & 255);
                        }
                        this.mLastSmooth = this.mSmoothPoint.get(this.speedIndex).floatValue();
                    }
                }
            }
            z = true;
        }
        if (z) {
            postInvalidate();
            this.mStartTime += this.intervalTime;
        }
        return z;
    }

    private boolean isEmpty(List list) {
        return list == null || list.size() == 0;
    }

    private boolean notAnArray(List list, int i) {
        return list != null && list.size() > i;
    }

    private double speed2PointY(int i) {
        try {
            return (this.height / 100.0f) * i;
        } catch (Exception e) {
            e.printStackTrace();
            return -1.0d;
        }
    }

    public float getBaseLineY(Paint paint) {
        return Math.abs(paint.ascent() + paint.descent()) / 2.0f;
    }

    public int getCurrentTime() {
        return this.mStartTime;
    }

    public String[] getData() {
        return this.mModeDataArr;
    }

    public boolean isRun() {
        return this.isRun;
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        try {
            if (this.mPoint.size() > 0) {
                this.mReversalPath.reset();
                this.mRunLinePath.reset();
                this.mVibrateRunLinePath.reset();
                this.mMaxPositionRunLinePath.reset();
                this.mMinPositionRunLinePath.reset();
                this.mSmoothRunLinePath.reset();
                LogUtil.e("mPoint.size():" + this.mPoint.size() + ", speedIndex:" + this.speedIndex);
                LogUtil.e("speed:" + this.mPoint.get(this.speedIndex));
                for (int i = 0; i <= this.speedIndex; i++) {
                    if (i == 0) {
                        float f = i;
                        this.mRunLinePath.moveTo(this.intervalX * f, this.height - ((this.mPoint.get(i).intValue() & 255) * this.intervalY));
                        if (notAnArray(this.mVibratePoint, i)) {
                            this.mVibrateRunLinePath.moveTo(this.intervalX * f, this.height - ((this.mVibratePoint.get(i).intValue() & 255) * this.intervalY));
                        }
                        if (notAnArray(this.mMaxPositionPoint, i)) {
                            this.mMaxPositionRunLinePath.moveTo(this.intervalX * f, this.height - ((this.mMaxPositionPoint.get(i).intValue() & 255) * this.intervalY));
                        }
                        if (notAnArray(this.mMinPositionPoint, i)) {
                            this.mMinPositionRunLinePath.moveTo(this.intervalX * f, this.height - ((this.mMinPositionPoint.get(i).intValue() & 255) * this.intervalY));
                        }
                        if (notAnArray(this.mSmoothPoint, i)) {
                            this.mSmoothRunLinePath.moveTo(this.intervalX * f, this.height - ((this.mSmoothPoint.get(i).intValue() & 255) * this.intervalY));
                        }
                    }
                    float f2 = i;
                    this.mRunLinePath.lineTo(this.intervalX * f2, this.height - ((this.mPoint.get(i).intValue() & 255) * this.intervalY));
                    this.mPoint.get(i).intValue();
                    if (notAnArray(this.mVibratePoint, i)) {
                        this.mVibrateRunLinePath.lineTo(this.intervalX * f2, this.height - ((this.mVibratePoint.get(i).intValue() & 255) * this.intervalY));
                    }
                    if (notAnArray(this.mMaxPositionPoint, i)) {
                        this.mMaxPositionRunLinePath.lineTo(this.intervalX * f2, this.height - ((this.mMaxPositionPoint.get(i).intValue() & 255) * this.intervalY));
                    }
                    if (notAnArray(this.mMinPositionPoint, i)) {
                        this.mMinPositionRunLinePath.lineTo(this.intervalX * f2, this.height - ((this.mMinPositionPoint.get(i).intValue() & 255) * this.intervalY));
                    }
                    if (notAnArray(this.mSmoothPoint, i)) {
                        this.mSmoothRunLinePath.lineTo(this.intervalX * f2, this.height - ((this.mSmoothPoint.get(i).intValue() & 255) * this.intervalY));
                    }
                }
                canvas.drawPath(this.mRunLinePath, this.mPlayPaint);
                canvas.drawPath(this.mVibrateRunLinePath, this.mVibratePlayPaint);
                canvas.drawPath(this.mMaxPositionRunLinePath, this.mMaxPositionPlayPaint);
                canvas.drawPath(this.mMinPositionRunLinePath, this.mMinPositionPlayPaint);
                canvas.drawPath(this.mSmoothRunLinePath, this.mSmoothPlayPaint);
                this.linePath.reset();
                this.vibrateLinePath.reset();
                this.maxPositionLinePath.reset();
                this.minPositionLinePath.reset();
                this.smoothLinePath.reset();
                for (int i2 = this.speedIndex; i2 < this.mPoint.size(); i2++) {
                    if (i2 == this.speedIndex) {
                        float f3 = i2;
                        this.linePath.moveTo(this.intervalX * f3, this.height - ((this.mPoint.get(i2).intValue() & 255) * this.intervalY));
                        if (notAnArray(this.mVibratePoint, i2)) {
                            this.vibrateLinePath.moveTo(this.intervalX * f3, this.height - ((this.mVibratePoint.get(i2).intValue() & 255) * this.intervalY));
                        }
                        if (notAnArray(this.mMaxPositionPoint, i2)) {
                            this.maxPositionLinePath.moveTo(this.intervalX * f3, this.height - ((this.mMaxPositionPoint.get(i2).intValue() & 255) * this.intervalY));
                        }
                        if (notAnArray(this.mMinPositionPoint, i2)) {
                            this.minPositionLinePath.moveTo(this.intervalX * f3, this.height - ((this.mMinPositionPoint.get(i2).intValue() & 255) * this.intervalY));
                        }
                        if (notAnArray(this.mSmoothPoint, i2)) {
                            this.smoothLinePath.moveTo(this.intervalX * f3, this.height - ((this.mSmoothPoint.get(i2).intValue() & 255) * this.intervalY));
                        }
                    }
                    float f4 = i2;
                    this.linePath.lineTo(this.intervalX * f4, this.height - ((this.mPoint.get(i2).intValue() & 255) * this.intervalY));
                    this.mPoint.get(i2).intValue();
                    if (notAnArray(this.mVibratePoint, i2)) {
                        this.vibrateLinePath.lineTo(this.intervalX * f4, this.height - ((this.mVibratePoint.get(i2).intValue() & 255) * this.intervalY));
                    }
                    if (notAnArray(this.mMaxPositionPoint, i2)) {
                        this.maxPositionLinePath.lineTo(this.intervalX * f4, this.height - ((this.mMaxPositionPoint.get(i2).intValue() & 255) * this.intervalY));
                    }
                    if (notAnArray(this.mMinPositionPoint, i2)) {
                        this.minPositionLinePath.lineTo(this.intervalX * f4, this.height - ((this.mMinPositionPoint.get(i2).intValue() & 255) * this.intervalY));
                    }
                    if (notAnArray(this.mSmoothPoint, i2)) {
                        this.smoothLinePath.lineTo(this.intervalX * f4, this.height - ((this.mSmoothPoint.get(i2).intValue() & 255) * this.intervalY));
                    }
                }
                canvas.drawPath(this.linePath, this.mPaint);
                canvas.drawPath(this.mReversalPath, this.mReversalPaint);
                canvas.drawPath(this.vibrateLinePath, this.mVibratePaint);
                canvas.drawPath(this.maxPositionLinePath, this.mMaxPositionPaint);
                canvas.drawPath(this.minPositionLinePath, this.mMinPositionPaint);
                canvas.drawPath(this.smoothLinePath, this.mSmoothPaint);
                float f5 = this.intervalX * this.speedIndex;
                float intValue = this.height - ((this.mPoint.get(r1).intValue() & 255) * this.intervalY);
                float f6 = this.radius;
                if (f5 < f6) {
                    f5 = f6;
                } else {
                    float f7 = this.width;
                    if (f5 > f7 - f6) {
                        f5 = f7 - f6;
                    }
                }
                float f8 = this.height;
                if (intValue > f8 - f6) {
                    intValue = f8 - f6;
                }
                canvas.drawCircle(f5, intValue, f6, this.mCirclePaint);
                if (notAnArray(this.mVibratePoint, this.speedIndex)) {
                    canvas.drawCircle(f5, this.height - ((this.mVibratePoint.get(this.speedIndex).intValue() & 255) * this.intervalY), this.radius, this.mVibrateCirclePaint);
                }
                if (notAnArray(this.mMaxPositionPoint, this.speedIndex)) {
                    canvas.drawCircle(f5, this.height - ((this.mMaxPositionPoint.get(this.speedIndex).intValue() & 255) * this.intervalY), this.radius, this.mMaxPositionCirclePaint);
                }
                if (notAnArray(this.mMinPositionPoint, this.speedIndex)) {
                    canvas.drawCircle(f5, this.height - ((this.mMinPositionPoint.get(this.speedIndex).intValue() & 255) * this.intervalY), this.radius, this.mMinPositionCirclePaint);
                }
                if (notAnArray(this.mSmoothPoint, this.speedIndex)) {
                    canvas.drawCircle(f5, this.height - ((this.mSmoothPoint.get(this.speedIndex).intValue() & 255) * this.intervalY), this.radius, this.mSmoothCirclePaint);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        this.width = getMeasuredWidth();
        LogUtil.e("centerX-radius width:" + this.width);
        this.height = getMeasuredHeight();
        float dip2px = MeasureUtil.dip2px(getContext(), 3.0f);
        this.radius = dip2px;
        this.intervalY = ((this.height - dip2px) - this.offsetY) / 100.0f;
        this.intervalX = this.width / this.totalPoint;
        if (this.mShader == null) {
            LinearGradient linearGradient = new LinearGradient(0.0f, 0.0f, this.width, 0.0f, new int[]{getResources().getColor(R.color.play_mode_line_left), getResources().getColor(R.color.play_mode_line_right)}, new float[]{0.0f, 1.0f}, Shader.TileMode.CLAMP);
            this.mShader = linearGradient;
            Paint paint = this.mPaint;
            if (paint != null) {
                paint.setShader(linearGradient);
            }
        }
        if (this.mVibrateShader == null) {
            LinearGradient linearGradient2 = new LinearGradient(0.0f, 0.0f, this.width, 0.0f, new int[]{getResources().getColor(R.color.play_mode_vibrate_line_left), getResources().getColor(R.color.play_mode_vibrate_line_right)}, new float[]{0.0f, 1.0f}, Shader.TileMode.CLAMP);
            this.mVibrateShader = linearGradient2;
            Paint paint2 = this.mVibratePaint;
            if (paint2 != null) {
                paint2.setShader(linearGradient2);
            }
        }
        if (this.mStartPositionShader == null) {
            LinearGradient linearGradient3 = new LinearGradient(0.0f, 0.0f, this.width, 0.0f, new int[]{getResources().getColor(R.color.play_mode_start_position_line_left), getResources().getColor(R.color.play_mode_start_position_line_right)}, new float[]{0.0f, 1.0f}, Shader.TileMode.CLAMP);
            this.mStartPositionShader = linearGradient3;
            Paint paint3 = this.mMinPositionPaint;
            if (paint3 != null) {
                paint3.setShader(linearGradient3);
            }
        }
        if (this.mStopPositionShader == null) {
            LinearGradient linearGradient4 = new LinearGradient(0.0f, 0.0f, this.width, 0.0f, new int[]{getResources().getColor(R.color.play_mode_stop_position_line_left), getResources().getColor(R.color.play_mode_stop_position_line_right)}, new float[]{0.0f, 1.0f}, Shader.TileMode.CLAMP);
            this.mStopPositionShader = linearGradient4;
            Paint paint4 = this.mMaxPositionPaint;
            if (paint4 != null) {
                paint4.setShader(linearGradient4);
            }
        }
        if (this.mSmoothShader == null) {
            LinearGradient linearGradient5 = new LinearGradient(0.0f, 0.0f, this.width, 0.0f, new int[]{getResources().getColor(R.color.play_mode_smooth_line_left), getResources().getColor(R.color.play_mode_smooth_line_right)}, new float[]{0.0f, 1.0f}, Shader.TileMode.CLAMP);
            this.mSmoothShader = linearGradient5;
            Paint paint5 = this.mSmoothPaint;
            if (paint5 != null) {
                paint5.setShader(linearGradient5);
            }
        }
    }

    public void setCurrentTime(int i) {
        this.mStartTime = i;
        if (this.isRun) {
            return;
        }
        initPoint();
    }

    public void setData(String str) {
        if (TextUtils.isEmpty(str)) {
            return;
        }
        this.mModeData = str;
        this.mModeDataArr = str.split("G");
    }

    public void setEndTime(int i) {
        this.mEndTime = i;
    }

    public void setOnPlayListener(OnPlayListener onPlayListener) {
        this.onPlayListener = onPlayListener;
    }

    public void startSync() {
        stopSync();
        this.isRun = true;
        Thread thread = new Thread() { // from class: com.hismith.hismithapp.view.LineChartPlayView3.1
            @Override // java.lang.Thread, java.lang.Runnable
            public void run() {
                while (LineChartPlayView3.this.isRun) {
                    try {
                        if (Thread.currentThread().isInterrupted()) {
                            break;
                        }
                        Thread.sleep(LineChartPlayView3.this.mPlayRefreshTime);
                        if (!LineChartPlayView3.this.initPoint()) {
                            break;
                        }
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                if (LineChartPlayView3.this.onPlayListener != null) {
                    LineChartPlayView3.this.onPlayListener.onPlaying(false);
                }
            }
        };
        this.mPlayThread = thread;
        thread.start();
        OnPlayListener onPlayListener = this.onPlayListener;
        if (onPlayListener != null) {
            onPlayListener.onPlaying(true);
        }
    }

    public void stopSync() {
        if (this.mPlayThread == null) {
            return;
        }
        this.isRun = false;
    }
}
