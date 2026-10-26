package com.hismith.hismithapp.adapter.callback;

import android.content.Context;
import android.os.Vibrator;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.ItemTouchHelper;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.adapter.PlayModeCreateAdapter;
import com.hismith.hismithapp.model.PlayModeItemModel;
import java.util.Collections;
import java.util.List;

/* loaded from: classes3.dex */
public class ItemDragCallback extends ItemTouchHelper.Callback {
    private PlayModeCreateAdapter mAdapter;
    private Context mContext;
    private List<PlayModeItemModel> mList;

    public ItemDragCallback(Context context, PlayModeCreateAdapter playModeCreateAdapter, List<PlayModeItemModel> list) {
        this.mContext = context;
        this.mAdapter = playModeCreateAdapter;
        this.mList = list;
    }

    @Override // androidx.recyclerview.widget.ItemTouchHelper.Callback
    public void clearView(RecyclerView recyclerView, RecyclerView.ViewHolder viewHolder) {
        super.clearView(recyclerView, viewHolder);
        this.mAdapter.initData();
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
        if (adapterPosition2 == 0 && "0".equals(this.mList.get(adapterPosition).getSingle())) {
            return false;
        }
        if (adapterPosition == 0 && this.mList.size() > 1 && "0".equals(this.mList.get(1).getSingle())) {
            return false;
        }
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
