package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.graphics.drawable.Drawable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.LineRoomInfoModel;
import com.hismith.hismithapp.view.CircleImageView;
import java.util.List;

/* loaded from: classes3.dex */
public class CircleImgAdapter extends RecyclerView.Adapter<ViewHolder> {
    private Context mContext;
    private List<LineRoomInfoModel.RoomInfoUserModel> mList;
    private OnItemClickListener onItemClickListener;

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private CircleImageView mCircleImg;
        private ImageView mImgCaptain;

        public ViewHolder(View view) {
            super(view);
            this.mImgCaptain = (ImageView) view.findViewById(R.id.item_circle_img_captain);
            this.mCircleImg = (CircleImageView) view.findViewById(R.id.item_circle_img);
        }
    }

    public CircleImgAdapter(Context context, List<LineRoomInfoModel.RoomInfoUserModel> list) {
        this.mContext = context;
        this.mList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final ViewHolder viewHolder, final int i) {
        Glide.with(this.mContext).load(this.mList.get(i).getUserHeadIcon()).into((RequestBuilder<Drawable>) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.adapter.CircleImgAdapter.1
            public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                viewHolder.mCircleImg.setImageDrawable(drawable);
            }

            @Override // com.bumptech.glide.request.target.Target
            public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
            }
        });
        if (this.mList.get(i).getCaptain() == 1) {
            viewHolder.mImgCaptain.setVisibility(0);
        } else {
            viewHolder.mImgCaptain.setVisibility(4);
        }
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CircleImgAdapter.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (CircleImgAdapter.this.onItemClickListener != null) {
                    CircleImgAdapter.this.onItemClickListener.onItemClick(i);
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_circle_img, viewGroup, false));
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
