package com.hismith.hismithapp.danmu;

import com.hismith.hismithapp.model.BarrageModel;
import master.flame.danmaku.danmaku.parser.IDataSource;

/* loaded from: classes3.dex */
public class BarrageSource implements IDataSource<BarrageModel> {
    private BarrageModel dataSource;

    public BarrageSource(BarrageModel barrageModel) {
        this.dataSource = barrageModel;
    }

    /* JADX WARN: Can't rename method to resolve collision */
    @Override // master.flame.danmaku.danmaku.parser.IDataSource
    public BarrageModel data() {
        return this.dataSource;
    }

    @Override // master.flame.danmaku.danmaku.parser.IDataSource
    public void release() {
    }
}
