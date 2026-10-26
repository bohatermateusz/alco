package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CompoundButton;
import androidx.appcompat.widget.AppCompatCheckBox;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import java.util.List;

/* loaded from: classes3.dex */
public class ItemCheckBoxAdapter extends RecyclerView.Adapter<MyViewHolder> {
    private Context mContext;
    private List<String> mList;
    private OnCheckItemListener onCheckItemListener;

    class MyViewHolder extends RecyclerView.ViewHolder {
        private AppCompatCheckBox check_item;

        public MyViewHolder(View view) {
            super(view);
            this.check_item = (AppCompatCheckBox) view.findViewById(R.id.check_item);
        }
    }

    public interface OnCheckItemListener {
        void onCheck(int i, boolean z);
    }

    public ItemCheckBoxAdapter(Context context, List<String> list, OnCheckItemListener onCheckItemListener) {
        this.mContext = context;
        this.mList = list;
        this.onCheckItemListener = onCheckItemListener;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(MyViewHolder myViewHolder, final int i) {
        myViewHolder.check_item.setText(this.mList.get(i));
        myViewHolder.check_item.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() { // from class: com.hismith.hismithapp.adapter.ItemCheckBoxAdapter.1
            @Override // android.widget.CompoundButton.OnCheckedChangeListener
            public void onCheckedChanged(CompoundButton compoundButton, boolean z) {
                if (ItemCheckBoxAdapter.this.onCheckItemListener != null) {
                    ItemCheckBoxAdapter.this.onCheckItemListener.onCheck(i, z);
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public MyViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new MyViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_checkbox, viewGroup, false));
    }

    public void setOnCheckItemListener(OnCheckItemListener onCheckItemListener) {
        this.onCheckItemListener = onCheckItemListener;
    }
}
