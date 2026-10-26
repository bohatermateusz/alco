package com.hismith.hismithapp.adapter.callback;

import android.content.Context;
import android.os.Vibrator;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.adapter.PiuTimerAdapter;
import com.hismith.hismithapp.manager.PiuTimerManager;
import com.hismith.hismithapp.model.PiuTimerModel;
import com.hismith.hismithapp.utils.LogUtil;
import java.util.Collections;
import java.util.List;

/* loaded from: classes3.dex */
public class PiuItemDragCallback extends ItemTouchHelper.Callback {
    private PiuTimerAdapter mAdapter;
    private Context mContext;
    private List<PiuTimerModel> mList;
    private int showPiuTimerMode;

    public PiuItemDragCallback(Context context, PiuTimerAdapter piuTimerAdapter, int i, List<PiuTimerModel> list) {
        this.mContext = context;
        this.mAdapter = piuTimerAdapter;
        this.showPiuTimerMode = i;
        this.mList = list;
    }

    @Override // androidx.recyclerview.widget.ItemTouchHelper.Callback
    public void clearView(RecyclerView recyclerView, RecyclerView.ViewHolder viewHolder) {
        super.clearView(recyclerView, viewHolder);
        this.mAdapter.notifyDataSetChanged();
    }

    @Override // androidx.recyclerview.widget.ItemTouchHelper.Callback
    public int getMovementFlags(RecyclerView recyclerView, RecyclerView.ViewHolder viewHolder) {
        return makeMovementFlags(recyclerView.getLayoutManager() instanceof GridLayoutManager ? 15 : recyclerView.getLayoutManager() instanceof LinearLayoutManager ? 3 : 0, 0);
    }

    @Override // androidx.recyclerview.widget.ItemTouchHelper.Callback
    public boolean isLongPressDragEnabled() {
        return true;
    }

    @Override // androidx.recyclerview.widget.ItemTouchHelper.Callback
    public boolean onMove(RecyclerView recyclerView, RecyclerView.ViewHolder viewHolder, RecyclerView.ViewHolder viewHolder2) {
        int adapterPosition = viewHolder.getAdapterPosition();
        int adapterPosition2 = viewHolder2.getAdapterPosition();
        if (adapterPosition < adapterPosition2) {
            int i = adapterPosition;
            while (i < adapterPosition2) {
                int i2 = i + 1;
                Collections.swap(this.mList, i, i2);
                i = i2;
            }
        } else {
            for (int i3 = adapterPosition; i3 > adapterPosition2; i3--) {
                Collections.swap(this.mList, i3, i3 - 1);
            }
        }
        LogUtil.e("textsssd Drag mList:" + this.mList.toString());
        PiuTimerManager.getInstance(this.mContext).saveShared(this.mList, this.showPiuTimerMode);
        if (this.showPiuTimerMode == PiuTimerManager.getInstance(this.mContext).getmPlayLocalMode()) {
            int playPosition = PiuTimerManager.getInstance(this.mContext).getPlayPosition();
            Context context = this.mContext;
            if (adapterPosition == playPosition) {
                PiuTimerManager.getInstance(context).setmPlayPosition(adapterPosition2);
            } else if (adapterPosition < PiuTimerManager.getInstance(context).getPlayPosition() && adapterPosition2 >= PiuTimerManager.getInstance(this.mContext).getPlayPosition()) {
                PiuTimerManager.getInstance(this.mContext).setmPlayPosition(PiuTimerManager.getInstance(this.mContext).getPlayPosition() - 1);
            } else if (adapterPosition > PiuTimerManager.getInstance(this.mContext).getPlayPosition() && adapterPosition2 <= PiuTimerManager.getInstance(this.mContext).getPlayPosition()) {
                PiuTimerManager.getInstance(this.mContext).setmPlayPosition(PiuTimerManager.getInstance(this.mContext).getPlayPosition() + 1);
            }
        }
        this.mAdapter.notifyItemMoved(adapterPosition, adapterPosition2);
        return true;
    }

    @Override // androidx.recyclerview.widget.ItemTouchHelper.Callback
    public void onSelectedChanged(RecyclerView.ViewHolder viewHolder, int i) {
        Vibrator vibrator;
        if (i != 0 && (vibrator = (Vibrator) this.mContext.getSystemService("vibrator")) != null) {
            vibrator.vibrate(70L);
        }
        super.onSelectedChanged(viewHolder, i);
    }

    @Override // androidx.recyclerview.widget.ItemTouchHelper.Callback
    public void onSwiped(RecyclerView.ViewHolder viewHolder, int i) {
    }
}
