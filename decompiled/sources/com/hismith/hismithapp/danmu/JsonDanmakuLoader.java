package com.hismith.hismithapp.danmu;

import com.hismith.hismithapp.model.BarrageModel;
import java.io.InputStream;
import master.flame.danmaku.danmaku.loader.ILoader;
import master.flame.danmaku.danmaku.loader.IllegalDataException;

/* loaded from: classes3.dex */
public class JsonDanmakuLoader implements ILoader {
    private static volatile JsonDanmakuLoader instance;
    private BarrageSource dataSource;

    private JsonDanmakuLoader() {
    }

    public static JsonDanmakuLoader instance() {
        if (instance == null) {
            synchronized (JsonDanmakuLoader.class) {
                if (instance == null) {
                    instance = new JsonDanmakuLoader();
                }
            }
        }
        return instance;
    }

    @Override // master.flame.danmaku.danmaku.loader.ILoader
    public BarrageSource getDataSource() {
        return this.dataSource;
    }

    public void load(BarrageModel barrageModel) {
        try {
            this.dataSource = new BarrageSource(barrageModel);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override // master.flame.danmaku.danmaku.loader.ILoader
    public void load(InputStream inputStream) throws IllegalDataException {
    }

    @Override // master.flame.danmaku.danmaku.loader.ILoader
    public void load(String str) throws IllegalDataException {
    }
}
