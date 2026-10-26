package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.LikeUserModel;
import com.hismith.hismithapp.view.CircleImageView;
import java.util.List;

/* loaded from: classes3.dex */
public class LikeAdapter extends RecyclerView.Adapter<ViewHolder> {
    private boolean isButton;
    private Context mContext;
    private List<LikeUserModel.User> mFanList;
    private OnItemCLickListener onItemCLickListener;
    private OnItemFollowClickListener onItemFollowClickListener;

    public interface OnItemCLickListener {
        void onItemClick(int i);
    }

    public interface OnItemFollowClickListener {
        void onItemFollow(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView mImgGender;
        private CircleImageView mImgHead;
        private TextView mTxtName;

        public ViewHolder(View view) {
            super(view);
            this.mImgHead = (CircleImageView) view.findViewById(R.id.fans_img_head);
            this.mImgGender = (ImageView) view.findViewById(R.id.fans_img_gender);
            this.mTxtName = (TextView) view.findViewById(R.id.fans_txt_name);
        }
    }

    public LikeAdapter(Context context, List<LikeUserModel.User> list) {
        this.isButton = false;
        this.mContext = context;
        this.mFanList = list;
    }

    public LikeAdapter(Context context, List<LikeUserModel.User> list, boolean z) {
        this.mContext = context;
        this.mFanList = list;
        this.isButton = z;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mFanList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        Glide.with(this.mContext).load(this.mFanList.get(i).getUserHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mImgHead);
        viewHolder.mImgHead.setKing(this.mFanList.get(i).getUserKing(), this.mContext, this.mFanList.get(i).getGradeIcon());
        if (this.mFanList.get(i).getUserGender() == 2) {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_transgender));
        } else if (this.mFanList.get(i).getUserGender() == 1) {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_men));
        } else {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_women));
        }
        viewHolder.mTxtName.setText(this.mFanList.get(i).getUserName());
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.LikeAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (LikeAdapter.this.onItemCLickListener != null) {
                    LikeAdapter.this.onItemCLickListener.onItemClick(i);
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_like_user, viewGroup, false));
    }

    public void setOnItemCLickListener(OnItemCLickListener onItemCLickListener) {
        this.onItemCLickListener = onItemCLickListener;
    }

    public void setOnItemFollowClickListener(OnItemFollowClickListener onItemFollowClickListener) {
        this.onItemFollowClickListener = onItemFollowClickListener;
    }
}
