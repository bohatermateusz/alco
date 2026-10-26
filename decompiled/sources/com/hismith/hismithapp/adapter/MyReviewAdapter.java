package com.hismith.hismithapp.adapter;

import android.content.Context;
import android.content.Intent;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.hismith.hismithapp.PersonalDetailsActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.TopicContentActivity;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ReviewDetailsDialog;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.IntentUtil;
import com.hismith.hismithapp.utils.StringUtils;
import com.hismith.hismithapp.view.CircleImageView;
import com.hismith.hismithapp.view.GridLayout;
import java.util.List;

/* loaded from: classes3.dex */
public class MyReviewAdapter extends RecyclerView.Adapter<ViewHolder> {
    private Context mContext;
    private List<TopicModel> mTopicModelList;
    private OnCollectClickListener onCollectClickListener;
    private OnDownClickListener onDownClickListener;
    private OnLikeClickListener onLikeClickListener;

    public interface OnCollectClickListener {
        void onCollectClick(int i);
    }

    public interface OnDownClickListener {
        void onDownClick(int i);
    }

    public interface OnLikeClickListener {
        void onLikeClick(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private GridLayout mReviewGridLayout;
        private ImageView mReviewImgCollect;
        private ImageView mReviewImgDown;
        private ImageView mReviewImgHead;
        private ImageView mReviewImgKing;
        private ImageView mReviewImgLike;
        private CircleImageView mReviewImgMyHead;
        private ImageView mReviewImgProgressCollect;
        private ImageView mReviewImgProgressLike;
        private ImageView mReviewImgRead;
        private ImageView mReviewImgReview;
        private LinearLayout mReviewLayoutAction;
        private LinearLayout mReviewLayoutCollect;
        private LinearLayout mReviewLayoutLike;
        private LinearLayout mReviewLayoutRead;
        private LinearLayout mReviewLayoutReview;
        private LinearLayout mReviewLayoutTopic;
        private LinearLayout mReviewLayoutUser;
        private TextView mReviewTxtCollect;
        private TextView mReviewTxtContent;
        private TextView mReviewTxtLike;
        private TextView mReviewTxtMyContent;
        private TextView mReviewTxtMyName;
        private TextView mReviewTxtMyTime;
        private TextView mReviewTxtName;
        private TextView mReviewTxtRead;
        private TextView mReviewTxtReview;
        private TextView mReviewTxtTitle;

        public ViewHolder(View view) {
            super(view);
            this.mReviewImgMyHead = (CircleImageView) view.findViewById(R.id.review_img_my_head);
            this.mReviewImgHead = (ImageView) view.findViewById(R.id.review_img_head);
            this.mReviewImgKing = (ImageView) view.findViewById(R.id.review_img_king);
            this.mReviewImgDown = (ImageView) view.findViewById(R.id.review_img_down);
            this.mReviewImgLike = (ImageView) view.findViewById(R.id.review_img_like);
            this.mReviewImgReview = (ImageView) view.findViewById(R.id.review_img_review);
            this.mReviewImgCollect = (ImageView) view.findViewById(R.id.review_img_collect);
            this.mReviewImgRead = (ImageView) view.findViewById(R.id.review_img_read);
            this.mReviewTxtMyName = (TextView) view.findViewById(R.id.review_txt_my_name);
            this.mReviewTxtMyTime = (TextView) view.findViewById(R.id.review_txt_my_time);
            this.mReviewTxtMyContent = (TextView) view.findViewById(R.id.review_txt_my_content);
            this.mReviewTxtName = (TextView) view.findViewById(R.id.review_txt_name);
            this.mReviewTxtTitle = (TextView) view.findViewById(R.id.review_txt_title);
            this.mReviewTxtContent = (TextView) view.findViewById(R.id.review_txt_content);
            this.mReviewTxtLike = (TextView) view.findViewById(R.id.review_txt_like);
            this.mReviewTxtReview = (TextView) view.findViewById(R.id.review_txt_review);
            this.mReviewTxtCollect = (TextView) view.findViewById(R.id.review_txt_collect);
            this.mReviewTxtRead = (TextView) view.findViewById(R.id.review_txt_read);
            this.mReviewLayoutTopic = (LinearLayout) view.findViewById(R.id.review_layout_topic);
            this.mReviewLayoutUser = (LinearLayout) view.findViewById(R.id.review_layout_user);
            this.mReviewLayoutAction = (LinearLayout) view.findViewById(R.id.review_layout_action);
            this.mReviewLayoutLike = (LinearLayout) view.findViewById(R.id.review_layout_like);
            this.mReviewLayoutReview = (LinearLayout) view.findViewById(R.id.review_layout_review);
            this.mReviewLayoutCollect = (LinearLayout) view.findViewById(R.id.review_layout_collect);
            this.mReviewLayoutRead = (LinearLayout) view.findViewById(R.id.review_layout_read);
            this.mReviewGridLayout = (GridLayout) view.findViewById(R.id.review_grid_layout);
            this.mReviewImgProgressLike = (ImageView) view.findViewById(R.id.review_img_progress_like);
            this.mReviewImgProgressCollect = (ImageView) view.findViewById(R.id.review_img_progress_collect);
        }
    }

    public MyReviewAdapter(Context context, List<TopicModel> list) {
        this.mContext = context;
        this.mTopicModelList = list;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.mTopicModelList.size();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(ViewHolder viewHolder, int i, List list) {
        onBindViewHolder2(viewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final ViewHolder viewHolder, final int i) {
        String str;
        String str2;
        String str3;
        int i2;
        if (MyApplication.getApplication().getmUserModel() != null) {
            str = MyApplication.getApplication().getmUserModel().getHeadIcon();
            i2 = MyApplication.getApplication().getmUserModel().getKing();
            str2 = MyApplication.getApplication().getmUserModel().getName();
            str3 = MyApplication.getApplication().getmUserModel().getGradeIcon();
        } else {
            str = "";
            str2 = "";
            str3 = str2;
            i2 = 0;
        }
        Glide.with(this.mContext).load(str).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mReviewImgMyHead);
        viewHolder.mReviewImgMyHead.setKing(i2, this.mContext, str3);
        viewHolder.mReviewImgMyHead.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.MyReviewAdapter.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                MyReviewAdapter.this.mContext.startActivity(new Intent(MyReviewAdapter.this.mContext, (Class<?>) PersonalDetailsActivity2.class));
            }
        });
        Glide.with(this.mContext).load(this.mTopicModelList.get(i).getUserHeadIcon()).placeholder(R.mipmap.ic_personal_head).into(viewHolder.mReviewImgHead);
        viewHolder.mReviewImgHead.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.MyReviewAdapter.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (((TopicModel) MyReviewAdapter.this.mTopicModelList.get(i)).getUserId() != null) {
                    IntentUtil.startUserInfoActivity(MyReviewAdapter.this.mContext, ((TopicModel) MyReviewAdapter.this.mTopicModelList.get(i)).getUserId().intValue());
                }
            }
        });
        if (this.mTopicModelList.get(i).getUserKing() == 1) {
            viewHolder.mReviewImgKing.setVisibility(0);
        } else {
            viewHolder.mReviewImgKing.setVisibility(8);
        }
        viewHolder.mReviewImgDown.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.MyReviewAdapter.3
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (MyReviewAdapter.this.onDownClickListener != null) {
                    MyReviewAdapter.this.onDownClickListener.onDownClick(i);
                }
            }
        });
        if (this.mTopicModelList.get(i).getLikeed().intValue() == 1) {
            viewHolder.mReviewImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
        } else {
            viewHolder.mReviewImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
        }
        if (this.mTopicModelList.get(i).getCollected().intValue() == 1) {
            viewHolder.mReviewImgCollect.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect_pressed));
        } else {
            viewHolder.mReviewImgCollect.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect));
        }
        if (this.mTopicModelList.get(i).getViewed().intValue() == 1) {
            viewHolder.mReviewImgRead.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_read_pressed));
        } else {
            viewHolder.mReviewImgRead.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_read));
        }
        if (this.mTopicModelList.get(i).getLikeStatus() == 1) {
            viewHolder.mReviewLayoutLike.setEnabled(false);
            viewHolder.mReviewImgLike.setVisibility(8);
            viewHolder.mReviewTxtLike.setVisibility(8);
            viewHolder.mReviewImgProgressLike.setVisibility(0);
            AnimUtil.rotate(viewHolder.mReviewImgProgressLike);
        } else {
            viewHolder.mReviewLayoutLike.setEnabled(true);
            viewHolder.mReviewImgLike.setVisibility(0);
            viewHolder.mReviewTxtLike.setVisibility(0);
            viewHolder.mReviewImgProgressLike.clearAnimation();
            viewHolder.mReviewImgProgressLike.setVisibility(8);
        }
        if (this.mTopicModelList.get(i).getCollectStatus() == 1) {
            viewHolder.mReviewLayoutCollect.setEnabled(false);
            viewHolder.mReviewImgCollect.setVisibility(8);
            viewHolder.mReviewTxtCollect.setVisibility(8);
            viewHolder.mReviewImgProgressCollect.setVisibility(0);
            AnimUtil.rotate(viewHolder.mReviewImgProgressCollect);
        } else {
            viewHolder.mReviewLayoutCollect.setEnabled(true);
            viewHolder.mReviewImgCollect.setVisibility(0);
            viewHolder.mReviewTxtCollect.setVisibility(0);
            viewHolder.mReviewImgProgressCollect.clearAnimation();
            viewHolder.mReviewImgProgressCollect.setVisibility(8);
        }
        if (this.mTopicModelList.get(i).getId() == null) {
            viewHolder.mReviewImgHead.setVisibility(8);
            viewHolder.mReviewTxtTitle.setVisibility(8);
            viewHolder.mReviewLayoutAction.setVisibility(8);
        } else {
            viewHolder.mReviewImgHead.setVisibility(0);
            viewHolder.mReviewTxtTitle.setVisibility(0);
            viewHolder.mReviewLayoutAction.setVisibility(0);
        }
        viewHolder.mReviewTxtMyName.setText(str2);
        viewHolder.mReviewTxtMyTime.setText(this.mTopicModelList.get(i).getReviewCreateDate());
        viewHolder.mReviewTxtMyContent.setText(this.mTopicModelList.get(i).getReviewContent());
        viewHolder.mReviewTxtName.setText(this.mTopicModelList.get(i).getUserName());
        viewHolder.mReviewTxtTitle.setText(this.mTopicModelList.get(i).getTitle());
        if (TextUtils.isEmpty(this.mTopicModelList.get(i).getContent())) {
            viewHolder.mReviewTxtContent.setVisibility(8);
        } else {
            viewHolder.mReviewTxtContent.setVisibility(0);
            viewHolder.mReviewTxtContent.setText(this.mTopicModelList.get(i).getContent());
        }
        viewHolder.mReviewTxtLike.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getLikeCount()));
        viewHolder.mReviewTxtReview.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getReviewCount()));
        viewHolder.mReviewTxtCollect.setVisibility(0);
        viewHolder.mReviewTxtCollect.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getCollectCount()));
        viewHolder.mReviewTxtRead.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getViewCount()));
        viewHolder.mReviewGridLayout.setAdapter(new CommAdapter<String>(this.mContext, this.mTopicModelList.get(i).getImages(), R.layout.item_topic_image) { // from class: com.hismith.hismithapp.adapter.MyReviewAdapter.4
            @Override // com.hismith.hismithapp.adapter.CommAdapter
            public void convert(XQuickViewHolder xQuickViewHolder, String str4, final int i3) {
                Glide.with(this.mContext).load(str4).placeholder(R.mipmap.hismith_image_loading).error(R.mipmap.hismith_image_load_err).into((ImageView) xQuickViewHolder.getView(R.id.item_topic_grid_image));
                xQuickViewHolder.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.MyReviewAdapter.4.1
                    @Override // android.view.View.OnClickListener
                    public void onClick(View view) {
                        ImagePreview.getInstance().setContext(AnonymousClass4.this.mContext).setTopicId(((TopicModel) MyReviewAdapter.this.mTopicModelList.get(i)).getId().intValue()).setIndex(i3).setImageList(((TopicModel) MyReviewAdapter.this.mTopicModelList.get(i)).getImages()).setFolderName("hismith/image").setEnableDragClose(true).start();
                    }
                });
            }
        });
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.MyReviewAdapter.5
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                int reviewMainId = ((TopicModel) MyReviewAdapter.this.mTopicModelList.get(i)).getReviewMainId();
                MyReviewAdapter myReviewAdapter = MyReviewAdapter.this;
                new ReviewDetailsDialog(MyReviewAdapter.this.mContext, Integer.valueOf(reviewMainId != 0 ? ((TopicModel) myReviewAdapter.mTopicModelList.get(i)).getReviewMainId() : ((TopicModel) myReviewAdapter.mTopicModelList.get(i)).getReviewId())).showDialog();
            }
        });
        viewHolder.mReviewLayoutTopic.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.MyReviewAdapter.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (((TopicModel) MyReviewAdapter.this.mTopicModelList.get(i)).getId() != null) {
                    Intent intent = new Intent(MyReviewAdapter.this.mContext, (Class<?>) TopicContentActivity.class);
                    intent.putExtra("topicId", ((TopicModel) MyReviewAdapter.this.mTopicModelList.get(i)).getId());
                    MyReviewAdapter.this.mContext.startActivity(intent);
                }
            }
        });
        viewHolder.mReviewLayoutLike.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.MyReviewAdapter.7
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (MyReviewAdapter.this.onLikeClickListener != null) {
                    viewHolder.mReviewLayoutLike.setEnabled(false);
                    viewHolder.mReviewImgLike.setVisibility(8);
                    viewHolder.mReviewTxtLike.setVisibility(8);
                    viewHolder.mReviewImgProgressLike.setVisibility(0);
                    AnimUtil.rotate(viewHolder.mReviewImgProgressLike);
                    ((TopicModel) MyReviewAdapter.this.mTopicModelList.get(i)).setLikeStatus(1);
                    MyReviewAdapter.this.onLikeClickListener.onLikeClick(i);
                }
            }
        });
        viewHolder.mReviewLayoutCollect.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.MyReviewAdapter.8
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (MyReviewAdapter.this.onCollectClickListener != null) {
                    viewHolder.mReviewLayoutCollect.setEnabled(false);
                    viewHolder.mReviewImgCollect.setVisibility(8);
                    viewHolder.mReviewTxtCollect.setVisibility(8);
                    viewHolder.mReviewImgProgressCollect.setVisibility(0);
                    AnimUtil.rotate(viewHolder.mReviewImgProgressCollect);
                    ((TopicModel) MyReviewAdapter.this.mTopicModelList.get(i)).setCollectStatus(1);
                    MyReviewAdapter.this.onCollectClickListener.onCollectClick(i);
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
        if ("like".equals(list.get(0))) {
            viewHolder.mReviewLayoutLike.setEnabled(true);
            viewHolder.mReviewImgProgressLike.clearAnimation();
            viewHolder.mReviewImgProgressLike.setVisibility(8);
            viewHolder.mReviewImgLike.setVisibility(0);
            viewHolder.mReviewTxtLike.setVisibility(0);
            viewHolder.mReviewTxtLike.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getLikeCount()));
            if (this.mTopicModelList.get(i).getLikeed().intValue() == 1) {
                viewHolder.mReviewImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
                return;
            } else {
                viewHolder.mReviewImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
                return;
            }
        }
        if ("collect".equals(list.get(0))) {
            viewHolder.mReviewLayoutCollect.setEnabled(true);
            viewHolder.mReviewImgProgressCollect.clearAnimation();
            viewHolder.mReviewImgProgressCollect.setVisibility(8);
            viewHolder.mReviewImgCollect.setVisibility(0);
            viewHolder.mReviewTxtCollect.setVisibility(0);
            viewHolder.mReviewTxtCollect.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getCollectCount()));
            if (this.mTopicModelList.get(i).getCollected().intValue() == 1) {
                viewHolder.mReviewImgCollect.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect_pressed));
            } else {
                viewHolder.mReviewImgCollect.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect));
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        return new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_review_my, viewGroup, false));
    }

    public void setOnCollectClickListener(OnCollectClickListener onCollectClickListener) {
        this.onCollectClickListener = onCollectClickListener;
    }

    public void setOnDownClickListener(OnDownClickListener onDownClickListener) {
        this.onDownClickListener = onDownClickListener;
    }

    public void setOnLikeClickListener(OnLikeClickListener onLikeClickListener) {
        this.onLikeClickListener = onLikeClickListener;
    }
}
