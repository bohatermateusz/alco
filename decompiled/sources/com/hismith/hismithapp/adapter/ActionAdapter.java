package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.ActionModel;
import java.util.List;

/* loaded from: classes3.dex */
public class ActionAdapter extends RecyclerView.Adapter<ViewHolder> {
    private Context mContext;
    private List<ActionModel> mList;
    private OnItemClickListener onItemClickListener;

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView mImgDot;
        private ImageView mImgLogo;
        private TextView mTxtName;

        public ViewHolder(View view) {
            super(view);
            this.mImgLogo = (ImageView) view.findViewById(R.id.item_action_img_logo);
            this.mTxtName = (TextView) view.findViewById(R.id.item_action_txt_name);
            this.mImgDot = (ImageView) view.findViewById(R.id.item_action_img_dot);
        }
    }

    public ActionAdapter(Context context, List<ActionModel> list) {
        this.mContext = context;
        this.mList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(ViewHolder viewHolder, int i, List list) {
        onBindViewHolder2(viewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        viewHolder.mImgLogo.setImageDrawable(this.mContext.getResources().getDrawable(this.mList.get(i).getLogoInt()));
        viewHolder.mTxtName.setText(this.mList.get(i).getName());
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.ActionAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (ActionAdapter.this.onItemClickListener != null) {
                    ActionAdapter.this.onItemClickListener.onItemClick(i);
                }
            }
        });
        if (this.mList.get(i).isNew()) {
            viewHolder.mImgDot.setVisibility(0);
        } else {
            viewHolder.mImgDot.setVisibility(8);
        }
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(ViewHolder viewHolder, int i, List<Object> list) {
        if (list.isEmpty()) {
            onBindViewHolder(viewHolder, i);
        } else if (this.mList.get(i).isNew()) {
            viewHolder.mImgDot.setVisibility(0);
        } else {
            viewHolder.mImgDot.setVisibility(8);
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_action, viewGroup, false));
    }

    public void setOnItemCLickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
