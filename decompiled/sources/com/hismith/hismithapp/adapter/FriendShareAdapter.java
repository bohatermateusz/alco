package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.model.UserSearchModel;
import com.hismith.hismithapp.view.CircleImageView;
import java.util.List;

/* loaded from: classes3.dex */
public class FriendShareAdapter extends RecyclerView.Adapter<ViewHolder> {
    private Context mContext;
    private List<UserSearchModel.User> mFriendList;
    private OnItemClickListener onItemClickListener;
    private int selectPosition = -1;

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView mImgGender;
        private CircleImageView mImgHead;
        private FrameLayout mLayoutSelect;
        private TextView mTxtName;

        public ViewHolder(View view) {
            super(view);
            this.mImgHead = (CircleImageView) view.findViewById(R.id.item_share_friend_img_head);
            this.mTxtName = (TextView) view.findViewById(R.id.item_share_friend_txt_name);
            this.mImgGender = (ImageView) view.findViewById(R.id.item_share_friend_img_gender);
            this.mLayoutSelect = (FrameLayout) view.findViewById(R.id.item_share_friend_layout_select);
        }
    }

    public FriendShareAdapter(Context context, List<UserSearchModel.User> list) {
        this.mContext = context;
        this.mFriendList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mFriendList.size();
    }

    public int getSelectPosition() {
        return this.selectPosition;
    }

    public List<UserSearchModel.User> getmFriendList() {
        return this.mFriendList;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        Glide.with(this.mContext).load(this.mFriendList.get(i).getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mImgHead);
        viewHolder.mImgHead.setKing(this.mFriendList.get(i).getKing(), this.mContext, this.mFriendList.get(i).getGradeIcon());
        if (this.mFriendList.get(i).getGender() == 2) {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_transgender));
        } else if (this.mFriendList.get(i).getGender() == 1) {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_men));
        } else {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_women));
        }
        viewHolder.mTxtName.setText(this.mFriendList.get(i).getName());
        if (this.selectPosition == i) {
            viewHolder.mLayoutSelect.setVisibility(0);
        } else {
            viewHolder.mLayoutSelect.setVisibility(8);
        }
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.FriendShareAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                FriendShareAdapter.this.selectPosition = i;
                FriendShareAdapter.this.notifyDataSetChanged();
                if (FriendShareAdapter.this.onItemClickListener != null) {
                    FriendShareAdapter.this.onItemClickListener.onItemClick(FriendShareAdapter.this.selectPosition);
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_friend_share, viewGroup, false));
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }

    public void setSelectPosition(int i) {
        this.selectPosition = i;
    }
}
