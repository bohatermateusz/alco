package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.ProductModel;
import java.util.List;

/* loaded from: classes3.dex */
public class IplayModeAdapter extends RecyclerView.Adapter<MyViewHolder> {
    private int clickPosition = -1;
    private Context mContext;
    private List<ProductModel.Mode> mModes;
    private OnItemClickListener onItemClickListener;

    class MyViewHolder extends RecyclerView.ViewHolder {
        private ImageView item_mode_bg;
        private ImageView mImgMode;
        private ImageView mLayoutMode;
        private ProgressBar mProgressMode;

        public MyViewHolder(View view) {
            super(view);
            this.mLayoutMode = (ImageView) view.findViewById(R.id.item_mode_layout_mode);
            this.item_mode_bg = (ImageView) view.findViewById(R.id.item_mode_bg);
            this.mImgMode = (ImageView) view.findViewById(R.id.item_mode_img_mode);
            this.mProgressMode = (ProgressBar) view.findViewById(R.id.item_mode_progress_mode);
        }
    }

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    public IplayModeAdapter(Context context, List<ProductModel.Mode> list) {
        this.mContext = context;
        this.mModes = list;
    }

    public int getClickPosition() {
        return this.clickPosition;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mModes.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(MyViewHolder myViewHolder, final int i) {
        Glide.with(this.mContext).load(Integer.valueOf(R.mipmap.ic_mode_select)).placeholder(R.mipmap.ic_mode_select).into(myViewHolder.mLayoutMode);
        if (this.mModes.get(i).getLocationIcon() > 0) {
            Glide.with(this.mContext).load(ContextCompat.getDrawable(this.mContext, this.mModes.get(i).getLocationIcon())).into(myViewHolder.mImgMode);
        } else if (!TextUtils.isEmpty(this.mModes.get(i).getmIcon())) {
            Glide.with(this.mContext).load(this.mModes.get(i).getmIcon()).into(myViewHolder.mImgMode);
        }
        if (this.clickPosition == i) {
            myViewHolder.mLayoutMode.setVisibility(0);
        } else {
            myViewHolder.mLayoutMode.setVisibility(4);
        }
        myViewHolder.item_mode_bg.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.IplayModeAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (IplayModeAdapter.this.onItemClickListener != null) {
                    IplayModeAdapter.this.clickPosition = i;
                    IplayModeAdapter.this.onItemClickListener.onItemClick(i);
                    IplayModeAdapter.this.notifyDataSetChanged();
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public MyViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new MyViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_iplay_mode, viewGroup, false));
    }

    public void setClickPosition(int i) {
        if (this.clickPosition != i) {
            this.clickPosition = i;
            notifyDataSetChanged();
        }
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
