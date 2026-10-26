package com.hismith.hismithapp.layoutmanager;

import androidx.recyclerview.widget.RecyclerView;

/* loaded from: classes3.dex */
public class HorizontalPageLayoutManager extends RecyclerView.LayoutManager implements PageDecorationLastJudge {
    @Override // androidx.recyclerview.widget.RecyclerView.LayoutManager
    public RecyclerView.LayoutParams generateDefaultLayoutParams() {
        return null;
    }

    @Override // com.hismith.hismithapp.layoutmanager.PageDecorationLastJudge
    public boolean isLastColumn(int i) {
        return false;
    }

    @Override // com.hismith.hismithapp.layoutmanager.PageDecorationLastJudge
    public boolean isLastRow(int i) {
        return false;
    }

    @Override // com.hismith.hismithapp.layoutmanager.PageDecorationLastJudge
    public boolean isPageLast(int i) {
        return false;
    }
}
