package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import java.util.List;

/* loaded from: classes3.dex */
public class PlaySMSModeAdapter extends RecyclerView.Adapter<MyViewHolder> {
    private int clickPosition = -1;
    private Context mContext;
    private List<String> mList;

    class MyViewHolder extends RecyclerView.ViewHolder {
        private ImageView item_mode_img_mode;
        private ImageView item_mode_layout_mode;

        public MyViewHolder(View view) {
            super(view);
            this.item_mode_layout_mode = (ImageView) view.findViewById(R.id.item_mode_layout_mode);
            this.item_mode_img_mode = (ImageView) view.findViewById(R.id.item_mode_img_mode);
        }
    }

    public PlaySMSModeAdapter(Context context, List<String> list) {
        this.mContext = context;
        this.mList = list;
    }

    public int getClickPosition() {
        return this.clickPosition;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(MyViewHolder myViewHolder, int i) {
        Glide.with(this.mContext).load(Integer.valueOf(R.mipmap.ic_mode_select)).placeholder(R.mipmap.ic_mode_select).into(myViewHolder.item_mode_layout_mode);
        Glide.with(this.mContext).load(this.mList.get(i)).into(myViewHolder.item_mode_img_mode);
        if (this.clickPosition == i) {
            myViewHolder.item_mode_layout_mode.setVisibility(0);
        } else {
            myViewHolder.item_mode_layout_mode.setVisibility(4);
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public MyViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new MyViewHolder(LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.item_hudong_sms_mode, viewGroup, false));
    }

    public void setClickPosition(int i) {
        if (this.clickPosition != i) {
            this.clickPosition = i;
            notifyDataSetChanged();
        }
    }
}
