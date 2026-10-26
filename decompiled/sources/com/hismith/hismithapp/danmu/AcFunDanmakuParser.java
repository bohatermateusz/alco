package com.hismith.hismithapp.danmu;

import androidx.core.view.ViewCompat;
import androidx.media3.exoplayer.upstream.CmcdData;
import com.hismith.hismithapp.utils.LogUtil;
import master.flame.danmaku.danmaku.model.BaseDanmaku;
import master.flame.danmaku.danmaku.model.IDanmakus;
import master.flame.danmaku.danmaku.model.android.Danmakus;
import master.flame.danmaku.danmaku.parser.BaseDanmakuParser;
import master.flame.danmaku.danmaku.parser.android.JSONSource;
import master.flame.danmaku.danmaku.util.DanmakuUtils;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/* loaded from: classes3.dex */
public class AcFunDanmakuParser extends BaseDanmakuParser {
    public AcFunDanmakuParser() {
        LogUtil.d("AcFunDanmakuParser: create");
    }

    private Danmakus doParse(JSONArray jSONArray) {
        LogUtil.d("doParse: " + jSONArray);
        Danmakus danmakus = new Danmakus();
        if (jSONArray != null && jSONArray.length() != 0) {
            for (int i = 0; i < jSONArray.length(); i++) {
                try {
                    JSONObject jSONObject = jSONArray.getJSONObject(i);
                    if (jSONObject != null) {
                        danmakus = parseJSONObject(jSONObject, danmakus);
                    }
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }
        }
        return danmakus;
    }

    private Danmakus parseJSONObject(JSONObject jSONObject, Danmakus danmakus) {
        int parseInt;
        if (danmakus == null) {
            danmakus = new Danmakus();
        }
        if (jSONObject != null && jSONObject.length() != 0) {
            for (int i = 0; i < jSONObject.length(); i++) {
                try {
                    String[] split = jSONObject.getString("c").split(",");
                    if (split.length > 0 && (parseInt = Integer.parseInt(split[2])) != 7) {
                        long parseFloat = (long) (Float.parseFloat(split[0]) * 1000.0f);
                        int parseInt2 = Integer.parseInt(split[1]);
                        int i2 = ViewCompat.MEASURED_STATE_MASK;
                        int i3 = parseInt2 | ViewCompat.MEASURED_STATE_MASK;
                        float parseFloat2 = Float.parseFloat(split[3]);
                        BaseDanmaku createDanmaku = this.mContext.mDanmakuFactory.createDanmaku(parseInt, this.mContext);
                        if (createDanmaku != null) {
                            createDanmaku.setTime(parseFloat);
                            createDanmaku.textSize = parseFloat2 * (this.mDispDensity - 0.6f);
                            createDanmaku.textColor = i3;
                            if (i3 <= -16777216) {
                                i2 = -1;
                            }
                            createDanmaku.textShadowColor = i2;
                            DanmakuUtils.fillText(createDanmaku, jSONObject.optString(CmcdData.OBJECT_TYPE_MANIFEST, "...."));
                            createDanmaku.index = i;
                            createDanmaku.setTimer(this.mTimer);
                            danmakus.addItem(createDanmaku);
                        }
                    }
                } catch (NumberFormatException e) {
                    LogUtil.d("parseJSONObject: " + e);
                } catch (JSONException e2) {
                    LogUtil.d("parseJSONObject: " + e2);
                }
            }
        }
        return danmakus;
    }

    @Override // master.flame.danmaku.danmaku.parser.BaseDanmakuParser
    public IDanmakus parse() {
        LogUtil.d("parse: IN");
        return (this.mDataSource == null || !(this.mDataSource instanceof JSONSource)) ? new Danmakus() : doParse(((JSONSource) this.mDataSource).data());
    }
}
