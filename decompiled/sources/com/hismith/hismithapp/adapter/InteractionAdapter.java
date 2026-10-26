package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.content.Intent;
import android.text.Html;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.TopicContentActivity;
import com.hismith.hismithapp.dialog.ReviewDetailsDialog;
import com.hismith.hismithapp.model.InteractiveModel;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.IntentUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.TimeUtil;
import com.hismith.hismithapp.view.CircleImageView;
import java.util.List;

/* loaded from: classes3.dex */
public class InteractionAdapter extends RecyclerView.Adapter<MyViewHolder> {
    private int LastInteractionId;
    private Context mContext;
    private List<InteractiveModel.Interactive> mList;
    private OnItemFollowClickListener onItemFollowClickListener;
    private final String TypeTopicLike = "topicLike";
    private final String TypeTopicCollect = "topicCollect";
    private final String TypeReview = "review";
    private final String TypeReviewLike = "reviewLike";
    private final String TypeReviewReply = "reviewReply";
    private final String TypeFollow = "follow";

    class MyViewHolder extends RecyclerView.ViewHolder {
        private TextView mBtnFollow;
        private ImageView mImgGender;
        private CircleImageView mImgHead;
        private ImageView mImgProgress;
        private FrameLayout mLayoutFollow;
        private View mLine;
        private TextView mTxtDate;
        private TextView mTxtLast;
        private TextView mTxtName;
        private TextView mTxtSourceContent;
        private TextView mTxtTypeNote;

        public MyViewHolder(View view) {
            super(view);
            this.mTxtDate = (TextView) view.findViewById(R.id.item_interaction_txt_date);
            this.mTxtName = (TextView) view.findViewById(R.id.item_interaction_txt_name);
            this.mTxtTypeNote = (TextView) view.findViewById(R.id.item_interaction_txt_typeNote);
            this.mTxtSourceContent = (TextView) view.findViewById(R.id.item_interaction_txt_sourceContent);
            this.mTxtLast = (TextView) view.findViewById(R.id.item_interaction_txt_last);
            this.mImgHead = (CircleImageView) view.findViewById(R.id.item_interaction_img_head);
            this.mImgGender = (ImageView) view.findViewById(R.id.item_interaction_img_gender);
            this.mBtnFollow = (TextView) view.findViewById(R.id.item_interaction_btn_follow);
            this.mLine = view.findViewById(R.id.item_interaction_line);
            this.mLayoutFollow = (FrameLayout) view.findViewById(R.id.item_interaction_layout_follow);
            this.mImgProgress = (ImageView) view.findViewById(R.id.item_interaction_img_progress);
            LanguageUtil.setText(InteractionAdapter.this.mContext, this.mTxtLast, "lastViewed");
        }
    }

    public interface OnItemFollowClickListener {
        void onItemFollow(int i);
    }

    public InteractionAdapter(Context context, List<InteractiveModel.Interactive> list, int i) {
        this.mContext = context;
        this.mList = list;
        this.LastInteractionId = i;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(MyViewHolder myViewHolder, int i, List list) {
        onBindViewHolder2(myViewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final MyViewHolder myViewHolder, final int i) {
        if (this.LastInteractionId == this.mList.get(i).getId()) {
            myViewHolder.mTxtLast.setVisibility(0);
        } else if (myViewHolder.mTxtLast.getVisibility() == 0) {
            myViewHolder.mTxtLast.setVisibility(8);
        }
        Glide.with(this.mContext).load(this.mList.get(i).getHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(myViewHolder.mImgHead);
        myViewHolder.mImgHead.setKing(this.mList.get(i).getKing(), this.mContext, this.mList.get(i).getGradeIcon());
        myViewHolder.mTxtDate.setText(TimeUtil.utc2Local(this.mList.get(i).getDate()));
        myViewHolder.mTxtName.setText(this.mList.get(i).getName());
        if ("2".equals(this.mList.get(i).getGender())) {
            myViewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_transgender));
        } else if ("1".equals(this.mList.get(i).getGender())) {
            myViewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_men));
        } else {
            myViewHolder.mImgGender.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_details_women));
        }
        myViewHolder.mLine.setVisibility(0);
        myViewHolder.mTxtSourceContent.setVisibility(0);
        myViewHolder.mLayoutFollow.setVisibility(8);
        myViewHolder.itemView.setOnClickListener(null);
        if (!TextUtils.isEmpty(this.mList.get(i).getType())) {
            String type = this.mList.get(i).getType();
            type.hashCode();
            switch (type) {
                case "topicCollect":
                case "topicLike":
                    String typeNote = this.mList.get(i).getTypeNote();
                    if (!TextUtils.isEmpty(typeNote)) {
                        myViewHolder.mTxtTypeNote.setText(Html.fromHtml(typeNote.replace("[", "<font color=\"" + this.mContext.getResources().getColor(R.color.app_style_color) + "\">").replace("]", "</font>")));
                    }
                    myViewHolder.mTxtSourceContent.setText(this.mList.get(i).getSourceContent());
                    myViewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.InteractionAdapter.1
                        @Override // android.view.View.OnClickListener
                        public void onClick(View view) {
                            try {
                                if (((InteractiveModel.Interactive) InteractionAdapter.this.mList.get(i)).getSourceDelete() == 1) {
                                    return;
                                }
                                int intValue = Integer.valueOf(((InteractiveModel.Interactive) InteractionAdapter.this.mList.get(i)).getSourceId()).intValue();
                                Intent intent = new Intent(InteractionAdapter.this.mContext, (Class<?>) TopicContentActivity.class);
                                intent.putExtra("topicId", intValue);
                                InteractionAdapter.this.mContext.startActivity(intent);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    });
                    break;
                case "reviewLike":
                    String typeNote2 = this.mList.get(i).getTypeNote();
                    if (!TextUtils.isEmpty(typeNote2)) {
                        myViewHolder.mTxtTypeNote.setText(Html.fromHtml(typeNote2.replace("[", "<font color=\"" + this.mContext.getResources().getColor(R.color.app_style_color) + "\">").replace("]", "</font>")));
                    }
                    myViewHolder.mTxtSourceContent.setText(this.mList.get(i).getSourceContent());
                    myViewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.InteractionAdapter.2
                        @Override // android.view.View.OnClickListener
                        public void onClick(View view) {
                            try {
                                if (((InteractiveModel.Interactive) InteractionAdapter.this.mList.get(i)).getSourceDelete() == 1) {
                                    return;
                                }
                                new ReviewDetailsDialog(InteractionAdapter.this.mContext, Integer.valueOf(Integer.valueOf(((InteractiveModel.Interactive) InteractionAdapter.this.mList.get(i)).getSourceId()).intValue()), true).showDialog();
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    });
                    break;
                case "follow":
                    String typeNote3 = this.mList.get(i).getTypeNote();
                    if (!TextUtils.isEmpty(typeNote3)) {
                        myViewHolder.mTxtTypeNote.setText(Html.fromHtml(typeNote3.replace("[", "<font color=\"" + this.mContext.getResources().getColor(R.color.app_style_color) + "\">").replace("]", "</font>")));
                    }
                    myViewHolder.mTxtSourceContent.setVisibility(8);
                    myViewHolder.mLayoutFollow.setVisibility(0);
                    int intValue = this.mList.get(i).getFollowed().intValue();
                    Context context = this.mContext;
                    if (intValue != 0) {
                        LanguageUtil.setText(context, myViewHolder.mBtnFollow, "followed");
                        myViewHolder.mLayoutFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_15));
                        myViewHolder.mBtnFollow.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
                        break;
                    } else {
                        myViewHolder.mBtnFollow.setText("+ " + LanguageUtil.getLanguageByKey(context, "followCount"));
                        myViewHolder.mLayoutFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_edit_style_15));
                        myViewHolder.mBtnFollow.setTextColor(this.mContext.getResources().getColor(R.color.white));
                        break;
                    }
                case "review":
                    myViewHolder.mTxtTypeNote.setText(this.mList.get(i).getResultContent());
                    myViewHolder.mTxtSourceContent.setText(this.mList.get(i).getSourceContent());
                    myViewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.InteractionAdapter.3
                        @Override // android.view.View.OnClickListener
                        public void onClick(View view) {
                            try {
                                if (((InteractiveModel.Interactive) InteractionAdapter.this.mList.get(i)).getSourceDelete() == 1) {
                                    return;
                                }
                                new ReviewDetailsDialog(InteractionAdapter.this.mContext, Integer.valueOf(Integer.valueOf(((InteractiveModel.Interactive) InteractionAdapter.this.mList.get(i)).getResultId()).intValue()), true).showDialog();
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    });
                    break;
                case "reviewReply":
                    myViewHolder.mTxtTypeNote.setText(this.mList.get(i).getResultContent());
                    myViewHolder.mTxtSourceContent.setText(this.mList.get(i).getSourceContent());
                    myViewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.InteractionAdapter.4
                        @Override // android.view.View.OnClickListener
                        public void onClick(View view) {
                            try {
                                if (((InteractiveModel.Interactive) InteractionAdapter.this.mList.get(i)).getSourceDelete() == 1) {
                                    return;
                                }
                                int intValue2 = Integer.valueOf(((InteractiveModel.Interactive) InteractionAdapter.this.mList.get(i)).getSourceId()).intValue();
                                new ReviewDetailsDialog(InteractionAdapter.this.mContext, Integer.valueOf(intValue2), Integer.valueOf(((InteractiveModel.Interactive) InteractionAdapter.this.mList.get(i)).getResultId()).intValue(), true).showDialog();
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    });
                    break;
            }
        }
        if (this.mList.get(i).getStatus() == 1) {
            myViewHolder.mLayoutFollow.setEnabled(false);
            myViewHolder.mBtnFollow.setVisibility(4);
            myViewHolder.mImgProgress.setVisibility(0);
            AnimUtil.rotate(myViewHolder.mImgProgress);
            myViewHolder.mLayoutFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_15));
        } else {
            myViewHolder.mLayoutFollow.setEnabled(true);
            myViewHolder.mBtnFollow.setVisibility(0);
            myViewHolder.mImgProgress.clearAnimation();
            myViewHolder.mImgProgress.setVisibility(4);
        }
        myViewHolder.mImgHead.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.InteractionAdapter.5
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                IntentUtil.startUserInfoActivity(InteractionAdapter.this.mContext, ((InteractiveModel.Interactive) InteractionAdapter.this.mList.get(i)).getUserId());
            }
        });
        myViewHolder.mLayoutFollow.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.InteractionAdapter.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (InteractionAdapter.this.onItemFollowClickListener != null) {
                    myViewHolder.mLayoutFollow.setEnabled(false);
                    myViewHolder.mBtnFollow.setVisibility(4);
                    myViewHolder.mLayoutFollow.setBackground(InteractionAdapter.this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_15));
                    myViewHolder.mImgProgress.setVisibility(0);
                    AnimUtil.rotate(myViewHolder.mImgProgress);
                    ((InteractiveModel.Interactive) InteractionAdapter.this.mList.get(i)).setStatus(1);
                    InteractionAdapter.this.onItemFollowClickListener.onItemFollow(i);
                }
            }
        });
    }

    /* renamed from: onBindViewHolder, reason: avoid collision after fix types in other method */
    public void onBindViewHolder2(MyViewHolder myViewHolder, int i, List<Object> list) {
        if (list.isEmpty()) {
            onBindViewHolder(myViewHolder, i);
            return;
        }
        myViewHolder.mLayoutFollow.setEnabled(true);
        myViewHolder.mBtnFollow.setVisibility(0);
        myViewHolder.mImgProgress.clearAnimation();
        myViewHolder.mImgProgress.setVisibility(4);
        int intValue = this.mList.get(i).getFollowed().intValue();
        Context context = this.mContext;
        if (intValue != 0) {
            LanguageUtil.setText(context, myViewHolder.mBtnFollow, "followed");
            myViewHolder.mLayoutFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_corners_border_style_15));
            myViewHolder.mBtnFollow.setTextColor(this.mContext.getResources().getColor(R.color.app_style_color));
        } else {
            myViewHolder.mBtnFollow.setText("+ " + LanguageUtil.getLanguageByKey(context, "followCount"));
            myViewHolder.mLayoutFollow.setBackground(this.mContext.getResources().getDrawable(R.drawable.bg_edit_style_15));
            myViewHolder.mBtnFollow.setTextColor(this.mContext.getResources().getColor(R.color.white));
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public MyViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new MyViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_interaction, viewGroup, false));
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onViewAttachedToWindow(MyViewHolder myViewHolder) {
        super.onViewAttachedToWindow((InteractionAdapter) myViewHolder);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onViewDetachedFromWindow(MyViewHolder myViewHolder) {
        super.onViewDetachedFromWindow((InteractionAdapter) myViewHolder);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onViewRecycled(MyViewHolder myViewHolder) {
        super.onViewRecycled((InteractionAdapter) myViewHolder);
    }

    public void setOnItemFollowClickListener(OnItemFollowClickListener onItemFollowClickListener) {
        this.onItemFollowClickListener = onItemFollowClickListener;
    }
}
