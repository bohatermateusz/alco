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
import com.hismith.hismithapp.model.UserSearchModel;
import com.hismith.hismithapp.view.CircleImageView;
import java.util.List;

/* loaded from: classes3.dex */
public class SearchFriendAdapter extends RecyclerView.Adapter<ViewHolder> {
    private Context mContext;
    private List<UserSearchModel.User> mFanList;
    private OnItemClickListener onItemClickListener;

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private ImageView mImgGender;
        private CircleImageView mImgHead;
        private ViewGroup mLayoutFollow;
        private TextView mTxtFollow;
        private TextView mTxtId;
        private TextView mTxtName;

        public ViewHolder(View view) {
            super(view);
            this.mImgHead = (CircleImageView) view.findViewById(R.id.fans_img_head);
            this.mImgGender = (ImageView) view.findViewById(R.id.fans_img_gender);
            this.mTxtName = (TextView) view.findViewById(R.id.fans_txt_name);
            this.mTxtId = (TextView) view.findViewById(R.id.fans_txt_id);
            this.mTxtFollow = (TextView) view.findViewById(R.id.fans_txt_follow);
            ViewGroup viewGroup = (ViewGroup) view.findViewById(R.id.fans_layout_follow);
            this.mLayoutFollow = viewGroup;
            viewGroup.setVisibility(8);
            this.mTxtFollow.setVisibility(8);
        }
    }

    public SearchFriendAdapter(Context context, List<UserSearchModel.User> list) {
        this.mContext = context;
        this.mFanList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mFanList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(ViewHolder viewHolder, final int i) {
        Glide.with(this.mContext).load(this.mFanList.get(i).getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mImgHead);
        viewHolder.mImgHead.setKing(this.mFanList.get(i).getKing(), this.mContext, this.mFanList.get(i).getGradeIcon());
        viewHolder.mImgHead.setIsOnline(this.mFanList.get(i).getOnline());
        if (this.mFanList.get(i).getGender() == 2) {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_transgender));
        } else if (this.mFanList.get(i).getGender() == 1) {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_men));
        } else {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_women));
        }
        viewHolder.mTxtName.setText(this.mFanList.get(i).getName());
        viewHolder.mTxtId.setText("ID: " + this.mFanList.get(i).getUserId());
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.SearchFriendAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (SearchFriendAdapter.this.onItemClickListener != null) {
                    SearchFriendAdapter.this.onItemClickListener.onItemClick(i);
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_fans, viewGroup, false));
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }
}
