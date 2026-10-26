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
import com.hismith.hismithapp.model.FansFocusModel;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.view.CircleImageView;
import java.util.List;

/* loaded from: classes3.dex */
public class FocusAdapter extends RecyclerView.Adapter<ViewHolder> {
    private boolean isButton;
    private Context mContext;
    private List<FansFocusModel.User> mFanList;
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
        private ImageView mImgProgress;
        private FrameLayout mLayoutFollow;
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
            this.mLayoutFollow = (FrameLayout) view.findViewById(R.id.fans_layout_follow);
            this.mImgProgress = (ImageView) view.findViewById(R.id.fans_img_progress);
        }
    }

    public FocusAdapter(Context context, List<FansFocusModel.User> list) {
        this.isButton = true;
        this.mContext = context;
        this.mFanList = list;
    }

    public FocusAdapter(Context context, List<FansFocusModel.User> list, boolean z) {
        this.mContext = context;
        this.mFanList = list;
        this.isButton = z;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mFanList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(ViewHolder viewHolder, int i, List list) {
        onBindViewHolder2(viewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final ViewHolder viewHolder, final int i) {
        Glide.with(this.mContext).load(this.mFanList.get(i).getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mImgHead);
        viewHolder.mImgHead.setKing(this.mFanList.get(i).getKing().intValue(), this.mContext, this.mFanList.get(i).getGradeIcon());
        viewHolder.mImgHead.setIsOnline(this.mFanList.get(i).getOnline());
        if (this.mFanList.get(i).getGender().intValue() == 2) {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_transgender));
        } else if (this.mFanList.get(i).getGender().intValue() == 1) {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_men));
        } else {
            viewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_women));
        }
        viewHolder.mTxtName.setText(this.mFanList.get(i).getName());
        viewHolder.mTxtId.setText("ID: " + this.mFanList.get(i).getUserId());
        if (this.isButton) {
            viewHolder.mLayoutFollow.setVisibility(0);
            LanguageUtil.setText(this.mContext, viewHolder.mTxtFollow, "followed");
            viewHolder.mLayoutFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_15));
            viewHolder.mTxtFollow.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
            if (this.mFanList.get(i).getFollowStatus() == 1) {
                viewHolder.mLayoutFollow.setEnabled(false);
                viewHolder.mTxtFollow.setVisibility(4);
                viewHolder.mImgProgress.setVisibility(0);
                AnimUtil.rotate(viewHolder.mImgProgress);
            } else {
                viewHolder.mLayoutFollow.setEnabled(true);
                viewHolder.mTxtFollow.setVisibility(0);
                viewHolder.mImgProgress.clearAnimation();
                viewHolder.mImgProgress.setVisibility(8);
            }
            viewHolder.mLayoutFollow.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.FocusAdapter.1
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    if (FocusAdapter.this.onItemFollowClickListener != null) {
                        viewHolder.mLayoutFollow.setEnabled(false);
                        viewHolder.mTxtFollow.setVisibility(4);
                        viewHolder.mLayoutFollow.setBackground(FocusAdapter.this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_15));
                        viewHolder.mImgProgress.setVisibility(0);
                        AnimUtil.rotate(viewHolder.mImgProgress);
                        ((FansFocusModel.User) FocusAdapter.this.mFanList.get(i)).setFollowStatus(1);
                        FocusAdapter.this.onItemFollowClickListener.onItemFollow(i);
                    }
                }
            });
        } else {
            viewHolder.mLayoutFollow.setVisibility(8);
        }
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.FocusAdapter.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (FocusAdapter.this.onItemCLickListener != null) {
                    FocusAdapter.this.onItemCLickListener.onItemClick(i);
                }
            }
        });
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(ViewHolder viewHolder, int i, List<Object> list) {
        if (list.isEmpty()) {
            onBindViewHolder(viewHolder, i);
            return;
        }
        if (this.isButton) {
            viewHolder.mLayoutFollow.setEnabled(true);
            viewHolder.mTxtFollow.setVisibility(0);
            viewHolder.mImgProgress.clearAnimation();
            viewHolder.mImgProgress.setVisibility(8);
            int IsFollow = this.mFanList.get(i).IsFollow();
            Context context = this.mContext;
            if (IsFollow != 0) {
                LanguageUtil.setText(context, viewHolder.mTxtFollow, "followed");
                viewHolder.mLayoutFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_15));
                viewHolder.mTxtFollow.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
            } else {
                viewHolder.mTxtFollow.setText("+ " + LanguageUtil.getLanguageByKey(context, "followCount"));
                viewHolder.mLayoutFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_edit_style_15));
                viewHolder.mTxtFollow.setTextColor(this.mContext.getResources().getColor(R.color.white));
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_fans, viewGroup, false));
    }

    public void setOnItemCLickListener(OnItemCLickListener onItemCLickListener) {
        this.onItemCLickListener = onItemCLickListener;
    }

    public void setOnItemFollowClickListener(OnItemFollowClickListener onItemFollowClickListener) {
        this.onItemFollowClickListener = onItemFollowClickListener;
    }
}
