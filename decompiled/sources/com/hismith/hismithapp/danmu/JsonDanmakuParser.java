package com.hismith.hismithapp.danmu;

import android.content.Context;
import android.text.TextUtils;
import androidx.core.view.ViewCompat;
import com.hismith.hismithapp.model.BarrageModel;
import com.hismith.hismithapp.utils.MeasureUtil;
import master.flame.danmaku.danmaku.model.BaseDanmaku;
import master.flame.danmaku.danmaku.model.DanmakuTimer;
import master.flame.danmaku.danmaku.model.IDanmakus;
import master.flame.danmaku.danmaku.model.android.Danmakus;
import master.flame.danmaku.danmaku.parser.BaseDanmakuParser;
import master.flame.danmaku.danmaku.util.DanmakuUtils;

/* loaded from: classes3.dex */
public class JsonDanmakuParser extends BaseDanmakuParser {
    private Context activityContext;
    private float mTextSize;

    public JsonDanmakuParser(Context context) {
        this.activityContext = context;
        new DanmakuTimer();
        this.mTextSize = MeasureUtil.sp2px(context, 14.0f);
    }

    public JsonDanmakuParser(Context context, float f) {
        this.activityContext = context;
        new DanmakuTimer();
        this.mTextSize = f;
    }

    /* JADX WARN: Removed duplicated region for block: B:19:0x0079  */
    /* JADX WARN: Removed duplicated region for block: B:25:0x009b A[SYNTHETIC] */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    private Danmakus doParse(BarrageModel barrageModel) {
        int i;
        Danmakus danmakus = new Danmakus();
        if (barrageModel != null && barrageModel.getDatas() != null && barrageModel.getDatas().length != 0) {
            for (int i2 = 0; i2 < barrageModel.getDatas().length; i2++) {
                String content = barrageModel.getDatas()[i2].getContent();
                if (!TextUtils.isEmpty(content)) {
                    BaseDanmaku createDanmaku = this.mContext.mDanmakuFactory.createDanmaku(1, this.mContext);
                    long second = (long) (barrageModel.getDatas()[i2].getSecond() * 1000.0d);
                    try {
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    if (!TextUtils.isEmpty(barrageModel.getDatas()[i2].getColor())) {
                        i = Integer.valueOf(barrageModel.getDatas()[i2].getColor(), 16).intValue() | ViewCompat.MEASURED_STATE_MASK;
                        if (createDanmaku == null) {
                            createDanmaku.setTime(second);
                            createDanmaku.textSize = this.mTextSize;
                            createDanmaku.textColor = i;
                            createDanmaku.textShadowColor = i > -16777216 ? -16777216 : -1;
                            DanmakuUtils.fillText(createDanmaku, content);
                            createDanmaku.index = i2;
                            createDanmaku.setTimer(this.mTimer);
                            createDanmaku.flags = this.mContext.mGlobalFlagValues;
                            danmakus.addItem(createDanmaku);
                        }
                    }
                    i = -1;
                    if (createDanmaku == null) {
                    }
                }
            }
        }
        return danmakus;
    }

    private float parseFloat(String str) {
        try {
            return Float.parseFloat(str);
        } catch (NumberFormatException unused) {
            return 0.0f;
        }
    }

    @Override // master.flame.danmaku.danmaku.parser.BaseDanmakuParser
    protected IDanmakus parse() {
        return (this.mDataSource == null || !(this.mDataSource instanceof BarrageSource)) ? new Danmakus() : doParse(((BarrageSource) this.mDataSource).data());
    }
}
