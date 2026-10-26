package com.hismith.hismithapp.view;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.util.Log;
import android.util.TypedValue;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ProgressBar;
import androidx.core.view.MotionEventCompat;
import com.google.gson.JsonObject;
import com.hismith.hismithapp.product.ProductUtil;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import kotlinx.coroutines.scheduling.WorkQueueKt;

/* loaded from: classes3.dex */
public class PlayView_Four extends View {
    private static final String TAG = "PlayView_Four";
    private Bitmap bitmap;
    private Canvas canvas;
    private int downSpeed;
    private int height;
    private boolean isEnable;
    private boolean isMy;
    private boolean isRun;
    private boolean isTouch;
    private Point lastPoint1;
    private Point lastPoint2;
    private float[] mDownYArr;
    private float[] mEndXs;
    private float[] mEndYs;
    private long mPlayRefreshTime;
    private Thread mPlayThread;
    private volatile PointList[] mPointArr;
    private ProgressBar mProgressBar;
    private BroadcastReceiver mReceiver;
    private float mScreenSpeed;
    private float[] mStartXs;
    private float[] mStartYs;
    private OnLineChangeListener onLineChangeListener;
    private OnMoveChangeListener onMoveChangeListener;
    private Paint paint;
    private Paint[] paintArr;
    private int pointRadius;
    private Random random;
    private float scale;
    private float scaleRadius;

    public interface OnLineChangeListener {
        void onLineChange(int i, JsonObject jsonObject);
    }

    public interface OnMoveChangeListener {
        void onMoveChange(int i);
    }

    class Point {
        public float radius;
        public int x;
        public int y;

        public Point(int i, int i2, float f) {
            this.x = i;
            this.y = i2;
            this.radius = f;
        }
    }

    class PointList {
        List<Point> mArray = new ArrayList();
        private final int maxNum = 35;

        PointList() {
        }

        public synchronized void add(int i, Point point) {
            this.mArray.add(i, point);
            if (this.mArray.size() > 35) {
                this.mArray.remove(r2.size() - 1);
            }
        }

        public synchronized void add(Point point) {
            this.mArray.add(point);
            if (this.mArray.size() > 35) {
                this.mArray.remove(0);
            }
        }

        public void clear() {
            this.mArray.clear();
        }

        public Point get(int i) {
            return this.mArray.get(i);
        }

        public void remove(int i) {
            this.mArray.remove(i);
        }

        public int size() {
            return this.mArray.size();
        }
    }

    public PlayView_Four(Context context) {
        this(context, null);
    }

    public PlayView_Four(Context context, AttributeSet attributeSet) {
        this(context, attributeSet, 0);
    }

    public PlayView_Four(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
        this.paintArr = new Paint[20];
        this.mDownYArr = new float[20];
        this.mPointArr = new PointList[20];
        this.mStartXs = new float[20];
        this.mStartYs = new float[20];
        this.mEndXs = new float[20];
        this.mEndYs = new float[20];
        this.height = 0;
        int dp2px = dp2px(20);
        this.pointRadius = dp2px;
        this.scale = 0.06f;
        this.scaleRadius = 0.06f * dp2px;
        this.mPlayRefreshTime = 33L;
        this.mScreenSpeed = 20.0f;
        this.downSpeed = 0;
        this.isTouch = true;
        this.isEnable = true;
        this.isRun = true;
        this.isMy = true;
        this.mReceiver = new BroadcastReceiver() { // from class: com.hismith.hismithapp.view.PlayView_Four.1
            @Override // android.content.BroadcastReceiver
            public void onReceive(Context context2, Intent intent) {
                String action = intent.getAction();
                if (Constans.RECEIVER_SLIDE_SPEED_ACTION.equals(action)) {
                    LogUtil.e("接收广播：com.hismith.hismithapp.action.slide_speed_action");
                    PlayView_Four.this.initScreenSpeed();
                } else if (Constans.RECEIVER_BLE_CONN_STATE.equals(action)) {
                    int intExtra = intent.getIntExtra("ble_state", 0);
                    if (intExtra == 2) {
                        PlayView_Four.this.initEnable();
                    } else if (intExtra == 0) {
                        PlayView_Four.this.isEnable = false;
                    }
                    LogUtil.e("接收广播：com.hismith.hismithapp.action.ble.conn.state, isEnable:" + PlayView_Four.this.isEnable + ", isMy:" + PlayView_Four.this.isMy);
                }
            }
        };
        init();
    }

    private boolean caculatePoints(float f, float f2, float f3, float f4, float f5, Point point, Point point2) {
        double pow = Math.pow(f, 2.0d) + 1.0d;
        float f6 = f2 - f4;
        double d = ((f * 2.0f) * f6) - (2.0f * f3);
        double sqrt = Math.sqrt(Math.pow(d, 2.0d) - ((4.0d * pow) * ((Math.pow(f3, 2.0d) + Math.pow(f6, 2.0d)) - Math.pow(f5, 2.0d))));
        double d2 = -d;
        double d3 = pow * 2.0d;
        point.x = (int) ((d2 + sqrt) / d3);
        point.y = (int) ((point.x * f) + f2);
        point2.x = (int) ((d2 - sqrt) / d3);
        point2.y = (int) ((f * point2.x) + f2);
        return true;
    }

    private int getRandomColor() {
        int nextInt = this.random.nextInt(256);
        int nextInt2 = this.random.nextInt(256);
        int i = nextInt + nextInt2;
        int i2 = 381 - i;
        if (i2 <= 0) {
            i2 = 0;
        }
        return Color.rgb(nextInt, nextInt2, i < 256 ? rand(WorkQueueKt.MASK, 255) : rand(i2, 255));
    }

    private void init() {
        for (int i = 0; i < this.mPointArr.length; i++) {
            this.mPointArr[i] = new PointList();
        }
        this.random = new Random();
        initEnable();
        registerReceiver();
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initEnable() {
        if (ProductUtil.isFType()) {
            this.isEnable = false;
        } else {
            this.isEnable = true;
        }
    }

    private Paint initPaint() {
        Paint paint = new Paint(4);
        paint.setStyle(Paint.Style.FILL);
        paint.setStrokeWidth(this.pointRadius);
        paint.setColor(getRandomColor());
        paint.setAntiAlias(true);
        paint.setFlags(1);
        paint.setDither(true);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeCap(Paint.Cap.ROUND);
        return paint;
    }

    private Path initPath(float f, float f2, float f3, float f4, float f5) {
        double d = f;
        double d2 = f5 / 2.0f;
        double d3 = (-1.0f) / ((f4 - f2) / (f3 - f));
        float cos = (float) (d - (Math.cos(Math.atan(d3)) * d2));
        double d4 = f2;
        float sin = (float) ((Math.sin(Math.atan(d3)) * d2) + d4);
        float cos2 = (float) (d + (Math.cos(Math.atan(d3)) * d2));
        float sin2 = (float) (d4 - (Math.sin(Math.atan(d3)) * d2));
        double d5 = f3;
        float cos3 = (float) (d5 - (Math.cos(Math.atan(d3)) * d2));
        double d6 = f4;
        float sin3 = (float) (d6 + (Math.sin(Math.atan(d3)) * d2));
        float cos4 = (float) (d5 + (Math.cos(Math.atan(d3)) * d2));
        float sin4 = (float) (d6 - (Math.sin(Math.atan(d3)) * d2));
        Path path = new Path();
        path.moveTo(cos, sin);
        path.lineTo(cos3, sin3);
        path.lineTo(cos4, sin4);
        path.lineTo(cos2, sin2);
        path.close();
        return path;
    }

    private int rand(int i, int i2) {
        return (this.random.nextInt(i2) % ((i2 - i) + 1)) + i;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void refreshRadius() {
        for (int i = 0; i < this.mPointArr.length; i++) {
            try {
                int i2 = 0;
                while (i2 < this.mPointArr[i].size()) {
                    if (i2 < this.mPointArr[i].size() && this.mPointArr[i].get(i2) != null) {
                        this.mPointArr[i].get(i2).radius -= this.scaleRadius;
                        if (this.mPointArr[i].get(i2).radius < 0.0f) {
                            this.mPointArr[i].remove(i2);
                        } else {
                            i2++;
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }
    }

    private void registerReceiver() {
        IntentFilter intentFilter = new IntentFilter();
        intentFilter.addAction(Constans.RECEIVER_SLIDE_SPEED_ACTION);
        intentFilter.addAction(Constans.RECEIVER_BLE_CONN_STATE);
        AppUtil.registerReceiver(getContext(), this.mReceiver, intentFilter);
    }

    private void startSync() {
        stopSync();
        this.isRun = true;
        LogUtil.e("startSync");
        Thread thread = new Thread() { // from class: com.hismith.hismithapp.view.PlayView_Four.2
            @Override // java.lang.Thread, java.lang.Runnable
            public void run() {
                while (PlayView_Four.this.isRun) {
                    try {
                        PlayView_Four.this.postInvalidate();
                        Thread.sleep(PlayView_Four.this.mPlayRefreshTime);
                        PlayView_Four.this.refreshRadius();
                    } catch (Exception e) {
                        e.printStackTrace();
                        LogUtil.e("循环报错");
                    }
                }
                LogUtil.e("循环停止");
            }
        };
        this.mPlayThread = thread;
        thread.start();
    }

    private void stopSync() {
        if (this.mPlayThread == null) {
            return;
        }
        this.isRun = false;
        this.mPlayThread = null;
        LogUtil.e("stopSync");
    }

    private void unregisterReceiver() {
        try {
            getContext().unregisterReceiver(this.mReceiver);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    protected int dp2px(int i) {
        return (int) TypedValue.applyDimension(1, i, getResources().getDisplayMetrics());
    }

    public float getmScreenSpeed() {
        return this.mScreenSpeed;
    }

    public void initScreenSpeed() {
        if (((Integer) SharedPreferencesUtil.get(getContext(), Constans.Shard_Slide_Speed_Position, -1)).intValue() > -1) {
            this.mScreenSpeed = Constans.Slide_Speed_Arr[r0];
        }
    }

    public boolean isMy() {
        return this.isMy;
    }

    public boolean isTouch() {
        return this.isTouch;
    }

    @Override // android.view.View
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        startSync();
    }

    @Override // android.view.View
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        stopSync();
        unregisterReceiver();
    }

    @Override // android.view.View
    protected void onDraw(Canvas canvas) {
        Canvas canvas2;
        super.onDraw(canvas);
        for (int i = 0; i < this.mPointArr.length; i++) {
            int i2 = 0;
            while (i2 < this.mPointArr[i].size()) {
                try {
                } catch (Exception e) {
                    e = e;
                    canvas2 = canvas;
                }
                if (this.paintArr[i] != null && this.mPointArr[i].get(i2) != null) {
                    this.paintArr[i].setStrokeWidth(this.mPointArr[i].get(i2).radius);
                    PointList[] pointListArr = this.mPointArr;
                    if (i2 == 0) {
                        canvas.drawPoint(pointListArr[i].get(i2).x, this.mPointArr[i].get(i2).y, this.paintArr[i]);
                    } else {
                        if (pointListArr[i].get(i2 - 1) != null && this.mPointArr[i].get(i2) != null) {
                            canvas2 = canvas;
                            try {
                                canvas2.drawLine(this.mPointArr[i].get(i2).x, this.mPointArr[i].get(i2).y, this.mPointArr[i].get(r4).x, this.mPointArr[i].get(r4).y, this.paintArr[i]);
                            } catch (Exception e2) {
                                e = e2;
                                e.printStackTrace();
                                i2++;
                                canvas = canvas2;
                            }
                            i2++;
                            canvas = canvas2;
                        }
                    }
                }
                canvas2 = canvas;
                i2++;
                canvas = canvas2;
            }
        }
    }

    @Override // android.view.View
    protected void onMeasure(int i, int i2) {
        super.onMeasure(i, i2);
        this.height = getMeasuredHeight();
    }

    public void onPointArrChange(int i, List<Integer> list) {
        JsonObject jsonObject = new JsonObject();
        for (int i2 = 0; list != null && i2 < list.size(); i2++) {
            JsonObject jsonObject2 = new JsonObject();
            jsonObject2.addProperty("x", Integer.valueOf(this.mPointArr[list.get(i2).intValue()].get(0).x));
            jsonObject2.addProperty("y", Integer.valueOf(this.mPointArr[list.get(i2).intValue()].get(0).y));
            jsonObject.add("L" + list.get(i2), jsonObject2);
        }
        OnLineChangeListener onLineChangeListener = this.onLineChangeListener;
        if (onLineChangeListener != null) {
            onLineChangeListener.onLineChange(i, jsonObject);
        }
    }

    @Override // android.view.View
    public boolean onTouchEvent(MotionEvent motionEvent) {
        if (!this.isEnable && this.isMy) {
            ToastUtil.showShortToastCenter(getContext(), LanguageUtil.getLanguageByKey(getContext(), "noConnectedDevice"));
            return false;
        }
        if (!this.isTouch) {
            return false;
        }
        int action = motionEvent.getAction() & 255;
        int action2 = (motionEvent.getAction() & MotionEventCompat.ACTION_POINTER_INDEX_MASK) >> 8;
        if (action == 0) {
            ProgressBar progressBar = this.mProgressBar;
            if (progressBar != null) {
                this.downSpeed = progressBar.getProgress();
            }
            for (int i = 0; i < motionEvent.getPointerCount(); i++) {
                int pointerId = motionEvent.getPointerId(i);
                float[] fArr = this.mDownYArr;
                if (pointerId < fArr.length) {
                    fArr[pointerId] = motionEvent.getY(i);
                    this.paintArr[pointerId] = initPaint();
                    this.mPointArr[pointerId].clear();
                    this.mPointArr[pointerId].add(0, new Point((int) motionEvent.getX(i), (int) motionEvent.getY(i), this.pointRadius));
                }
            }
        } else if (action == 1) {
            Log.d(TAG, "MotionEvent.ACTION_UP");
        } else if (action == 2) {
            float f = this.downSpeed;
            ArrayList arrayList = new ArrayList();
            for (int i2 = 0; i2 < motionEvent.getPointerCount(); i2++) {
                int pointerId2 = motionEvent.getPointerId(i2);
                if (pointerId2 < this.mDownYArr.length) {
                    arrayList.add(Integer.valueOf(pointerId2));
                    this.mPointArr[pointerId2].add(0, new Point((int) motionEvent.getX(i2), (int) motionEvent.getY(i2), this.pointRadius));
                    f += ((this.mDownYArr[pointerId2] - motionEvent.getY(i2)) / this.height) * this.mScreenSpeed;
                }
            }
            int i3 = (int) f;
            onPointArrChange(i3, arrayList);
            OnMoveChangeListener onMoveChangeListener = this.onMoveChangeListener;
            if (onMoveChangeListener != null) {
                onMoveChangeListener.onMoveChange(i3);
            }
        } else if (action == 5) {
            for (int i4 = 0; i4 < motionEvent.getPointerCount(); i4++) {
                int pointerId3 = motionEvent.getPointerId(i4);
                if (i4 != 0) {
                    float[] fArr2 = this.mDownYArr;
                    if (pointerId3 < fArr2.length) {
                        fArr2[pointerId3] = motionEvent.getY(i4);
                        this.paintArr[pointerId3] = initPaint();
                        this.mPointArr[pointerId3].clear();
                        this.mPointArr[pointerId3].add(0, new Point((int) motionEvent.getX(i4), (int) motionEvent.getY(i4), this.pointRadius));
                    }
                }
            }
        } else if (action == 6) {
            int pointerId4 = motionEvent.getPointerId(action2);
            LogUtil.e("ACTION_POINTER_UP index:" + pointerId4);
            float y = this.mDownYArr[pointerId4] - motionEvent.getY(action2);
            float f2 = (y / this.height) * this.mScreenSpeed;
            Log.d(TAG, "第" + (pointerId4 + 1) + "个手指抬起 moveY:" + y + ", upMoveSpeed:" + f2);
            this.downSpeed = (int) (this.downSpeed + f2);
        }
        return true;
    }

    public void setMy(boolean z) {
        this.isMy = z;
    }

    public void setOnLineChangeListener(OnLineChangeListener onLineChangeListener) {
        this.onLineChangeListener = onLineChangeListener;
    }

    public void setOnMoveChangeListener(OnMoveChangeListener onMoveChangeListener) {
        this.onMoveChangeListener = onMoveChangeListener;
    }

    public void setPointArr(Map<String, android.graphics.Point> map) {
        if (map == null) {
            return;
        }
        try {
            for (String str : map.keySet()) {
                int intValue = str.length() >= 2 ? Integer.valueOf(str.substring(1, 2)).intValue() : 0;
                android.graphics.Point point = map.get(str);
                if (this.mPointArr[intValue].size() == 0) {
                    this.paintArr[intValue] = initPaint();
                }
                this.mPointArr[intValue].add(0, new Point(point.x, point.y, this.pointRadius));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void setProgressBar(ProgressBar progressBar) {
        this.mProgressBar = progressBar;
    }

    public void setTouch(boolean z) {
        this.isTouch = z;
    }

    public void setmScreenSpeed(float f) {
        int intValue = ((Integer) SharedPreferencesUtil.get(getContext(), Constans.Shard_Slide_Speed_Position, -1)).intValue();
        if (intValue > -1) {
            f = Constans.Slide_Speed_Arr[intValue];
        }
        this.mScreenSpeed = f;
    }
}
