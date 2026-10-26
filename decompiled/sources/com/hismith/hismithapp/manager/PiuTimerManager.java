package com.hismith.hismithapp.manager;

import android.content.Context;
import android.content.Intent;
import android.os.Handler;
import android.text.TextUtils;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.hismith.hismithapp.model.PiuTimerModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/* loaded from: classes3.dex */
public class PiuTimerManager {
    private static PiuTimerManager instance;
    private Gson gson;
    private Context mContext;
    private List<PiuTimerModel> mList;
    private int mRandomMaxTime;
    private int mRandomMinTime;
    private int mPlayLocalMode = 1;
    private int mPlayPosition = -1;
    private int mProgress = 0;
    private boolean isPlay = false;
    private Handler mHandler = new Handler();

    public PiuTimerManager(Context context) {
        this.mContext = context;
        clear();
        refreshTimer();
    }

    private String getDefaultRandomTime() {
        ArrayList arrayList = new ArrayList();
        arrayList.add(new PiuTimerModel(0));
        arrayList.add(new PiuTimerModel(10));
        return this.gson.toJson(arrayList);
    }

    public static PiuTimerManager getInstance(Context context) {
        PiuTimerManager piuTimerManager;
        synchronized (PiuTimerManager.class) {
            piuTimerManager = instance;
            if (piuTimerManager == null) {
                piuTimerManager = new PiuTimerManager(context);
                instance = piuTimerManager;
            }
        }
        return piuTimerManager;
    }

    private int randomIntSmart(int i, int i2) {
        return ThreadLocalRandom.current().nextInt(Math.min(i, i2), Math.max(i, i2) + 1);
    }

    public void clear() {
        this.isPlay = false;
        this.mPlayLocalMode = 1;
        this.mPlayPosition = -1;
        this.mHandler.removeCallbacksAndMessages(null);
        refreshTimer();
        LogUtil.e("clear() mPlayLocalMode:" + this.mPlayLocalMode + ", mPlayPosition:" + this.mPlayPosition);
    }

    public void deleteShared(List<PiuTimerModel> list, int i, int i2) {
        String json = this.gson.toJson(list);
        if (i == 1) {
            SharedPreferencesUtil.put(this.mContext, Constans.localPiuTimer, json);
        } else if (i == 2) {
            SharedPreferencesUtil.put(this.mContext, Constans.localPiuTimer2, json);
        } else if (i == 3) {
            SharedPreferencesUtil.put(this.mContext, Constans.localPiuTimer3, json);
        }
        if (this.mPlayLocalMode == i) {
            this.mList = list;
            int i3 = this.mPlayPosition;
            if (i3 == i2) {
                this.mPlayPosition = i3 - 1;
                nextTimer();
            } else if (i3 > i2) {
                this.mPlayPosition = i3 - 1;
            }
        }
    }

    public int getPlayPosition() {
        return this.mPlayPosition;
    }

    public List<PiuTimerModel> getmList(int i) {
        String str;
        List<PiuTimerModel> arrayList = new ArrayList<>();
        this.gson = new Gson();
        if (i == 1) {
            str = (String) SharedPreferencesUtil.get(this.mContext, Constans.localPiuTimer);
        } else if (i == 2) {
            str = (String) SharedPreferencesUtil.get(this.mContext, Constans.localPiuTimer2);
        } else if (i == 3) {
            str = (String) SharedPreferencesUtil.get(this.mContext, Constans.localPiuTimer3);
        } else if (i != 4) {
            str = "";
        } else {
            str = (String) SharedPreferencesUtil.get(this.mContext, Constans.randomPiuTimer4);
            if (TextUtils.isEmpty(str)) {
                str = getDefaultRandomTime();
            }
        }
        if (!TextUtils.isEmpty(str)) {
            arrayList = (List) this.gson.fromJson(str, new TypeToken<List<PiuTimerModel>>() { // from class: com.hismith.hismithapp.manager.PiuTimerManager.1
            }.getType());
        }
        LogUtil.e("getmList list:" + str);
        return arrayList;
    }

    public int getmPlayLocalMode() {
        return this.mPlayLocalMode;
    }

    public int getmProgress() {
        return this.mProgress;
    }

    public boolean isPlay() {
        return this.isPlay;
    }

    public void nextTimer() {
        int i = this.mPlayPosition + 1;
        this.mPlayPosition = i;
        this.mProgress = 0;
        if (i >= this.mList.size()) {
            this.mPlayPosition = 0;
        }
        LogUtil.e("nextTimer isPlay:" + this.isPlay);
        if (this.isPlay) {
            this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.manager.PiuTimerManager.3
                @Override // java.lang.Runnable
                public void run() {
                    PiuTimerManager.this.playTimer();
                }
            }, 100L);
        }
    }

    public void pauseTimer() {
        if (this.isPlay) {
            Intent intent = new Intent(Constans.RECEIVER_PIU_TIMER_ACTION);
            intent.putExtra(Constans.EXTRA_PIU_TIMER_ACTION, Constans.TIMER_ACTION_PAUSE);
            this.mContext.sendBroadcast(intent);
            this.isPlay = false;
            this.mHandler.removeCallbacksAndMessages(null);
        }
    }

    public boolean playPositionTimer(int i, int i2) {
        this.mPlayPosition = i;
        this.mPlayLocalMode = i2;
        this.mProgress = 0;
        refreshTimer();
        return playTimer();
    }

    public boolean playTimer() {
        List<PiuTimerModel> list = this.mList;
        if (list == null || list.isEmpty() || this.mPlayPosition >= this.mList.size()) {
            stopTimer();
            return false;
        }
        if (this.mPlayPosition <= -1) {
            this.mPlayPosition = 0;
        }
        LogUtil.e("playTimer " + this.mPlayPosition + " list:" + this.mList.toString());
        try {
            int i = this.mPlayLocalMode;
            List<PiuTimerModel> list2 = this.mList;
            PiuTimerModel piuTimerModel = i == 4 ? new PiuTimerModel(randomIntSmart(list2.get(0).getSecond(), this.mList.get(1).getSecond())) : list2.get(this.mPlayPosition);
            Intent intent = new Intent(Constans.RECEIVER_PIU_TIMER_ACTION);
            intent.putExtra(Constans.EXTRA_PIU_TIMER_ACTION, Constans.TIMER_ACTION_PLAY);
            intent.putExtra(Constans.EXTRA_PIU_TIMER_MODEL, piuTimerModel);
            this.mContext.sendBroadcast(intent);
            this.isPlay = true;
            return true;
        } catch (Exception e) {
            e.printStackTrace();
            stopTimer();
            return false;
        }
    }

    public void refreshTimer() {
        String str;
        this.gson = new Gson();
        int i = this.mPlayLocalMode;
        if (i == 1) {
            str = (String) SharedPreferencesUtil.get(this.mContext, Constans.localPiuTimer);
        } else if (i == 2) {
            str = (String) SharedPreferencesUtil.get(this.mContext, Constans.localPiuTimer2);
        } else if (i == 3) {
            str = (String) SharedPreferencesUtil.get(this.mContext, Constans.localPiuTimer3);
        } else if (i != 4) {
            str = "";
        } else {
            str = (String) SharedPreferencesUtil.get(this.mContext, Constans.randomPiuTimer4);
            if (TextUtils.isEmpty(str)) {
                str = getDefaultRandomTime();
            }
        }
        if (TextUtils.isEmpty(str)) {
            this.mList = new ArrayList();
        } else {
            this.mList = (List) this.gson.fromJson(str, new TypeToken<List<PiuTimerModel>>() { // from class: com.hismith.hismithapp.manager.PiuTimerManager.2
            }.getType());
        }
        LogUtil.e("refreshTimer list:" + this.mList.toString());
    }

    public void saveShared(List<PiuTimerModel> list, int i) {
        LogUtil.e("saveShared list:" + list.toString());
        String json = this.gson.toJson(list);
        if (i == 1) {
            SharedPreferencesUtil.put(this.mContext, Constans.localPiuTimer, json);
        } else if (i == 2) {
            SharedPreferencesUtil.put(this.mContext, Constans.localPiuTimer2, json);
        } else if (i == 3) {
            SharedPreferencesUtil.put(this.mContext, Constans.localPiuTimer3, json);
        } else if (i == 4) {
            SharedPreferencesUtil.put(this.mContext, Constans.randomPiuTimer4, json);
        }
        if (this.mPlayLocalMode == i) {
            this.mList = list;
            LogUtil.e("saveShared mPlayLocalMode(" + this.mPlayLocalMode + ") = localPiuTimerMode(" + i + ")");
        }
    }

    public void setProgress(int i) {
        this.mProgress = i;
    }

    public void setmPlayLocalMode(int i) {
        this.mPlayLocalMode = i;
        refreshTimer();
    }

    public void setmPlayPosition(int i) {
        this.mPlayPosition = i;
    }

    public void stopTimer() {
        if (this.isPlay) {
            Intent intent = new Intent(Constans.RECEIVER_PIU_TIMER_ACTION);
            intent.putExtra(Constans.EXTRA_PIU_TIMER_ACTION, Constans.TIMER_ACTION_STOP);
            this.mContext.sendBroadcast(intent);
            this.isPlay = false;
            this.mPlayPosition = -1;
            this.mHandler.removeCallbacksAndMessages(null);
        }
    }

    public void switchPlay() {
        boolean z = this.isPlay;
        this.isPlay = !z;
        if (z) {
            pauseTimer();
        } else {
            playTimer();
        }
    }
}
