package com.hismith.hismithapp.adapter;

import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.fragment.app.FragmentActivity;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.PersonalDetailsActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.TopicContentActivity;
import com.hismith.hismithapp.UserDetailsActivity2;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ReportSelectDialog;
import com.hismith.hismithapp.dialog.TopicActionDialog;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.model.TopicActionModel;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.utils.AnimUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.DialogUtil;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.StringUtils;
import com.hismith.hismithapp.utils.TimeUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.hismith.hismithapp.view.GridLayout;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.List;

/* loaded from: classes3.dex */
public class CommunityAdapter extends RecyclerView.Adapter<ViewHolder> {
    private final int VIEW_TYPE_DATA;
    private final int VIEW_TYPE_EMPTY;
    private RetrofitRequest baseRequest;
    private boolean isUserClick;
    private BaseActivity mContext;
    private List<TopicModel> mTopicModelList;
    private OnItemClickListener onItemClickListener;
    private OnUserInfoListener onUserInfoListener;
    private boolean showEmptyView;

    /* renamed from: com.hismith.hismithapp.adapter.CommunityAdapter$3, reason: invalid class name */
    class AnonymousClass3 implements View.OnClickListener {
        final /* synthetic */ int val$position;

        AnonymousClass3(int i) {
            this.val$position = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            if (((TopicModel) CommunityAdapter.this.mTopicModelList.get(this.val$position)).getUserId() != null) {
                TopicActionDialog topicActionDialog = new TopicActionDialog(CommunityAdapter.this.mContext);
                topicActionDialog.setOnClickItemActionListener(new TopicActionDialog.OnClickItemActionListener() { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.3.1
                    @Override // com.hismith.hismithapp.dialog.TopicActionDialog.OnClickItemActionListener
                    public void btnClickDelete(final int i) {
                        DialogUtil.showConfirmDialog(CommunityAdapter.this.mContext, "prompt", "deleteTopic", null, null, new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.3.1.3
                            @Override // android.view.View.OnClickListener
                            public void onClick(View view2) {
                                CommunityAdapter.this.topicDelete(i);
                                DialogUtil.dismissDialog();
                            }
                        });
                    }

                    @Override // com.hismith.hismithapp.dialog.TopicActionDialog.OnClickItemActionListener
                    public void btnClickNotInterested(final int i) {
                        DialogUtil.showConfirmDialog(CommunityAdapter.this.mContext, "prompt", "notInterestedTopic", null, "confirm", new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.3.1.2
                            @Override // android.view.View.OnClickListener
                            public void onClick(View view2) {
                                CommunityAdapter.this.topicUninterestedAction(i);
                                DialogUtil.dismissDialog();
                            }
                        });
                    }

                    @Override // com.hismith.hismithapp.dialog.TopicActionDialog.OnClickItemActionListener
                    public void btnClickReport(final int i) {
                        new ReportSelectDialog(CommunityAdapter.this.mContext) { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.3.1.1
                            @Override // com.hismith.hismithapp.dialog.ReportSelectDialog
                            public void btnSubmit(String str, String str2) {
                                CommunityAdapter.this.complaint(i, "topic", str, str2);
                            }
                        }.showDialog();
                    }
                }, this.val$position);
                if (((TopicModel) CommunityAdapter.this.mTopicModelList.get(this.val$position)).getUserId().equals(Integer.valueOf(MyApplication.getApplication().getmUserModel().getUserId()))) {
                    topicActionDialog.showDelete().initDelete().show();
                } else {
                    topicActionDialog.hideDelete().initDelete().show();
                }
            }
        }
    }

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    private class OnUserInfoClickListener implements View.OnClickListener {
        private int mPosition;

        public OnUserInfoClickListener(int i) {
            this.mPosition = i;
        }

        @Override // android.view.View.OnClickListener
        public void onClick(View view) {
            if (CommunityAdapter.this.onUserInfoListener != null && ((TopicModel) CommunityAdapter.this.mTopicModelList.get(this.mPosition)).getUserId() != null) {
                CommunityAdapter.this.onUserInfoListener.onUserInfoClick(((TopicModel) CommunityAdapter.this.mTopicModelList.get(this.mPosition)).getUserId().intValue());
                return;
            }
            if (((TopicModel) CommunityAdapter.this.mTopicModelList.get(this.mPosition)).getUserId() != null) {
                if (MyApplication.getApplication().getmUserModel() != null && ((TopicModel) CommunityAdapter.this.mTopicModelList.get(this.mPosition)).getUserId().intValue() == MyApplication.getApplication().getmUserModel().getUserId()) {
                    CommunityAdapter.this.mContext.startActivity(new Intent(CommunityAdapter.this.mContext, (Class<?>) PersonalDetailsActivity2.class));
                } else {
                    Intent intent = new Intent(CommunityAdapter.this.mContext, (Class<?>) UserDetailsActivity2.class);
                    intent.putExtra("userId", ((TopicModel) CommunityAdapter.this.mTopicModelList.get(this.mPosition)).getUserId());
                    CommunityAdapter.this.mContext.startActivity(intent);
                }
            }
        }
    }

    public interface OnUserInfoListener {
        void onUserInfoClick(int i);
    }

    class ViewHolder extends RecyclerView.ViewHolder {
        private TextView mEmptyTxt1;
        private GridLayout mTopicGridLayout;
        private ImageView mTopicImgDown;
        private ImageView mTopicImgFavorite;
        private CircleImageView mTopicImgHead;
        private ImageView mTopicImgLike;
        private ImageView mTopicImgProgressFavorite;
        private ImageView mTopicImgProgressLike;
        private ImageView mTopicImgRead;
        private ImageView mTopicImgReview;
        private LinearLayout mTopicLayoutFavorite;
        private LinearLayout mTopicLayoutLike;
        private TextView mTopicTxtContent;
        private TextView mTopicTxtFavorite;
        private TextView mTopicTxtLike;
        private TextView mTopicTxtName;
        private TextView mTopicTxtRead;
        private TextView mTopicTxtReview;
        private TextView mTopicTxtTime;
        private TextView mTopicTxtTitle;

        public ViewHolder(View view) {
            super(view);
            this.mTopicImgHead = (CircleImageView) view.findViewById(R.id.topic_img_head);
            this.mTopicImgDown = (ImageView) view.findViewById(R.id.topic_img_down);
            this.mTopicImgLike = (ImageView) view.findViewById(R.id.topic_img_like);
            this.mTopicImgReview = (ImageView) view.findViewById(R.id.topic_img_review);
            this.mTopicImgFavorite = (ImageView) view.findViewById(R.id.topic_img_favorite);
            this.mTopicImgRead = (ImageView) view.findViewById(R.id.topic_img_read);
            this.mTopicTxtName = (TextView) view.findViewById(R.id.topic_txt_name);
            this.mTopicTxtTime = (TextView) view.findViewById(R.id.topic_txt_time);
            this.mTopicTxtTitle = (TextView) view.findViewById(R.id.topic_txt_title);
            this.mTopicTxtContent = (TextView) view.findViewById(R.id.topic_txt_content);
            this.mTopicTxtLike = (TextView) view.findViewById(R.id.topic_txt_like);
            this.mTopicTxtReview = (TextView) view.findViewById(R.id.topic_txt_review);
            this.mTopicTxtFavorite = (TextView) view.findViewById(R.id.topic_txt_favorite);
            this.mTopicTxtRead = (TextView) view.findViewById(R.id.topic_txt_read);
            this.mTopicGridLayout = (GridLayout) view.findViewById(R.id.topic_grid_layout);
            this.mTopicLayoutLike = (LinearLayout) view.findViewById(R.id.topic_layout_like);
            this.mTopicLayoutFavorite = (LinearLayout) view.findViewById(R.id.topic_layout_favorite);
            this.mTopicImgProgressLike = (ImageView) view.findViewById(R.id.topic_img_progress_like);
            this.mTopicImgProgressFavorite = (ImageView) view.findViewById(R.id.topic_img_progress_favorite);
            this.mEmptyTxt1 = (TextView) view.findViewById(R.id.empty_txt1);
        }
    }

    public CommunityAdapter(BaseActivity baseActivity, List<TopicModel> list) {
        this.isUserClick = true;
        this.VIEW_TYPE_EMPTY = 1;
        this.VIEW_TYPE_DATA = 2;
        this.showEmptyView = false;
        this.mContext = baseActivity;
        this.mTopicModelList = list;
        this.baseRequest = new RetrofitRequest();
    }

    public CommunityAdapter(BaseActivity baseActivity, List<TopicModel> list, boolean z) {
        this.isUserClick = true;
        this.VIEW_TYPE_EMPTY = 1;
        this.VIEW_TYPE_DATA = 2;
        this.showEmptyView = false;
        this.mContext = baseActivity;
        this.mTopicModelList = list;
        this.baseRequest = new RetrofitRequest();
        this.isUserClick = z;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void complaint(int i, String str, String str2, String str3) {
        Observable<HttpModel> observeOn = this.baseRequest.getApi().complaint((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mTopicModelList.get(i).getId(), str, str2, str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.mContext;
        observeOn.subscribe(new RxSubscribe<HttpModel>(baseActivity, baseActivity.progressDialog, true) { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.11
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(CommunityAdapter.this.mContext, LanguageUtil.getLanguageByKey(CommunityAdapter.this.mContext, "reportedSuccess"));
                }
            }
        });
    }

    private View getEmptyView(ViewGroup viewGroup) {
        LogUtil.e("空数据getEmptyView");
        return LayoutInflater.from(this.mContext).inflate(R.layout.layout_empty, viewGroup, false);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicCollectAction(final int i, Integer num, final String str) {
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), num, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicActionModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.9
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i)).setCollectStatus(0);
                CommunityAdapter.this.notifyItemChanged(i, "collect");
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i)).setCollectStatus(0);
                CommunityAdapter.this.notifyItemChanged(i, "collect");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicActionModel topicActionModel) {
                if (topicActionModel.getSuccess() == 1) {
                    String collectCount = topicActionModel.getCollectCount();
                    if ("collect".equals(str)) {
                        ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i)).setCollected(1);
                    } else if ("uncollect".equals(str)) {
                        ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i)).setCollected(0);
                    }
                    ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i)).setCollectCount(collectCount);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicDelete(final int i) {
        Observable<HttpModel> observeOn = this.baseRequest.getApi().topicDel((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mTopicModelList.get(i).getId()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.mContext;
        observeOn.subscribe(new RxSubscribe<HttpModel>(baseActivity, baseActivity.progressDialog) { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.7
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    CommunityAdapter.this.mTopicModelList.remove(i);
                    ToastUtil.showShortToastCenter(CommunityAdapter.this.mContext, LanguageUtil.getLanguageByKey(CommunityAdapter.this.mContext, "deletedMsg"));
                    CommunityAdapter.this.notifyItemRemoved(i);
                    CommunityAdapter communityAdapter = CommunityAdapter.this;
                    communityAdapter.notifyItemRangeChanged(i, communityAdapter.mTopicModelList.size());
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicLikeAction(final int i, Integer num, final String str) {
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), num, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicActionModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.8
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i)).setLikeStatus(0);
                CommunityAdapter.this.notifyItemChanged(i, "like");
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i)).setLikeStatus(0);
                CommunityAdapter.this.notifyItemChanged(i, "like");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicActionModel topicActionModel) {
                if (topicActionModel.getSuccess() == 1) {
                    String likeCount = topicActionModel.getLikeCount();
                    if ("like".equals(str)) {
                        ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i)).setLikeed(1);
                    } else if ("unlike".equals(str)) {
                        ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i)).setLikeed(0);
                    }
                    ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i)).setLikeCount(likeCount);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicUninterestedAction(final int i) {
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mTopicModelList.get(i).getId(), "uninterested").subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.10
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    LogUtil.e("不感兴趣话题position:" + i + ", id:" + ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i)).getId());
                    CommunityAdapter.this.mTopicModelList.remove(i);
                    CommunityAdapter.this.notifyItemRemoved(i);
                    CommunityAdapter communityAdapter = CommunityAdapter.this;
                    communityAdapter.notifyItemRangeChanged(i, communityAdapter.mTopicModelList.size());
                }
            }
        });
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        List<TopicModel> list = this.mTopicModelList;
        int size = list != null ? list.size() : 0;
        return size > 0 ? size : this.showEmptyView ? 1 : 0;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemViewType(int i) {
        return isEmptyPosition(i) ? 1 : 2;
    }

    public boolean isEmptyPosition(int i) {
        List<TopicModel> list = this.mTopicModelList;
        return i == 0 && this.showEmptyView && (list != null ? list.size() : 0) == 0;
    }

    public boolean isShowEmptyView() {
        return this.showEmptyView;
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public /* bridge */ /* synthetic */ void onBindViewHolder(ViewHolder viewHolder, int i, List list) {
        onBindViewHolder2(viewHolder, i, (List<Object>) list);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public void onBindViewHolder(final ViewHolder viewHolder, int i) {
        CommunityAdapter communityAdapter;
        final int i2;
        if (isEmptyPosition(i)) {
            LogUtil.e("空数据onBindViewHolder" + i);
            if (viewHolder.mEmptyTxt1 != null) {
                LanguageUtil.setText(this.mContext, viewHolder.mEmptyTxt1, "noTopicData");
                return;
            }
            return;
        }
        LogUtil.e("CommunityAdapter:position:" + i + ", 内容：" + this.mTopicModelList.get(i).toString());
        viewHolder.mTopicTxtName.setText(this.mTopicModelList.get(i).getUserName());
        viewHolder.mTopicTxtTime.setText(TimeUtil.utc2Local(this.mTopicModelList.get(i).getCreateDate()));
        viewHolder.mTopicTxtTitle.setText(this.mTopicModelList.get(i).getTitle());
        if (TextUtils.isEmpty(this.mTopicModelList.get(i).getContent())) {
            viewHolder.mTopicTxtContent.setVisibility(8);
        } else {
            viewHolder.mTopicTxtContent.setVisibility(0);
            viewHolder.mTopicTxtContent.setText(this.mTopicModelList.get(i).getContent().trim());
        }
        viewHolder.mTopicTxtLike.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getLikeCount()));
        viewHolder.mTopicTxtReview.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getReviewCount()));
        viewHolder.mTopicTxtFavorite.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getCollectCount()));
        viewHolder.mTopicTxtRead.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getViewCount()));
        if (this.mTopicModelList.get(i).getLikeed().intValue() == 1) {
            viewHolder.mTopicImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
        } else {
            viewHolder.mTopicImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
        }
        if (this.mTopicModelList.get(i).getCollected().intValue() == 1) {
            viewHolder.mTopicImgFavorite.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect_pressed));
        } else {
            viewHolder.mTopicImgFavorite.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect));
        }
        if (this.mTopicModelList.get(i).getViewed().intValue() == 1) {
            viewHolder.mTopicImgRead.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_read_pressed));
        } else {
            viewHolder.mTopicImgRead.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_read));
        }
        viewHolder.mTopicImgHead.setImageResource(R.mipmap.ic_personal_head);
        Glide.with((FragmentActivity) this.mContext).load(this.mTopicModelList.get(i).getUserHeadIcon()).placeholder(R.mipmap.ic_personal_head).into((RequestBuilder) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.1
            public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                viewHolder.mTopicImgHead.setImageDrawable(drawable);
            }

            @Override // com.bumptech.glide.request.target.Target
            public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
            }
        });
        viewHolder.mTopicImgHead.setKing(this.mTopicModelList.get(i).getUserKing(), this.mContext, this.mTopicModelList.get(i).getGradeIcon());
        if (this.mTopicModelList.get(i).getImages().size() > 0) {
            viewHolder.mTopicGridLayout.setVisibility(0);
            communityAdapter = this;
            i2 = i;
            viewHolder.mTopicGridLayout.setAdapter(new CommAdapter<String>(this.mContext, this.mTopicModelList.get(i).getImages(), R.layout.item_topic_image) { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.2
                @Override // com.hismith.hismithapp.adapter.CommAdapter
                public void convert(XQuickViewHolder xQuickViewHolder, String str, final int i3) {
                    Glide.with(this.mContext).load(str).placeholder(R.mipmap.hismith_image_loading).error(R.mipmap.hismith_image_load_err).into((ImageView) xQuickViewHolder.getView(R.id.item_topic_grid_image));
                    xQuickViewHolder.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.2.1
                        @Override // android.view.View.OnClickListener
                        public void onClick(View view) {
                            ImagePreview.getInstance().setContext(AnonymousClass2.this.mContext).setTopicId(((TopicModel) CommunityAdapter.this.mTopicModelList.get(i2)).getId().intValue()).setIndex(i3).setImageList(((TopicModel) CommunityAdapter.this.mTopicModelList.get(i2)).getImages()).setFolderName("hismith/image").setEnableDragClose(true).start();
                        }
                    });
                }
            });
        } else {
            communityAdapter = this;
            i2 = i;
            viewHolder.mTopicGridLayout.setVisibility(8);
        }
        if (communityAdapter.isUserClick) {
            OnUserInfoClickListener onUserInfoClickListener = communityAdapter.new OnUserInfoClickListener(i2);
            viewHolder.mTopicImgHead.setOnClickListener(onUserInfoClickListener);
            viewHolder.mTopicTxtName.setOnClickListener(onUserInfoClickListener);
        }
        viewHolder.mTopicImgDown.setOnClickListener(communityAdapter.new AnonymousClass3(i2));
        if (communityAdapter.mTopicModelList.get(i2).getLikeStatus() == 1) {
            viewHolder.mTopicLayoutLike.setEnabled(false);
            viewHolder.mTopicLayoutLike.setVisibility(4);
            viewHolder.mTopicImgProgressLike.setVisibility(0);
            AnimUtil.rotate(viewHolder.mTopicImgProgressLike);
        } else {
            viewHolder.mTopicLayoutLike.setEnabled(true);
            viewHolder.mTopicLayoutLike.setVisibility(0);
            viewHolder.mTopicImgProgressLike.clearAnimation();
            viewHolder.mTopicImgProgressLike.setVisibility(8);
        }
        if (communityAdapter.mTopicModelList.get(i2).getCollectStatus() == 1) {
            viewHolder.mTopicLayoutFavorite.setEnabled(false);
            viewHolder.mTopicLayoutFavorite.setVisibility(4);
            viewHolder.mTopicImgProgressFavorite.setVisibility(0);
            AnimUtil.rotate(viewHolder.mTopicImgProgressFavorite);
        } else {
            viewHolder.mTopicLayoutFavorite.setEnabled(true);
            viewHolder.mTopicLayoutFavorite.setVisibility(0);
            viewHolder.mTopicImgProgressFavorite.clearAnimation();
            viewHolder.mTopicImgProgressFavorite.setVisibility(8);
        }
        viewHolder.mTopicLayoutLike.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.4
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (((TopicModel) CommunityAdapter.this.mTopicModelList.get(i2)).getId() != null) {
                    viewHolder.mTopicLayoutLike.setEnabled(false);
                    viewHolder.mTopicLayoutLike.setVisibility(4);
                    viewHolder.mTopicImgProgressLike.setVisibility(0);
                    AnimUtil.rotate(viewHolder.mTopicImgProgressLike);
                    String str = ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i2)).getLikeed().intValue() == 1 ? "unlike" : "like";
                    ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i2)).setLikeStatus(1);
                    CommunityAdapter communityAdapter2 = CommunityAdapter.this;
                    communityAdapter2.topicLikeAction(i2, ((TopicModel) communityAdapter2.mTopicModelList.get(i2)).getId(), str);
                }
            }
        });
        viewHolder.mTopicLayoutFavorite.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.5
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (((TopicModel) CommunityAdapter.this.mTopicModelList.get(i2)).getId() != null) {
                    viewHolder.mTopicLayoutFavorite.setEnabled(false);
                    viewHolder.mTopicLayoutFavorite.setVisibility(4);
                    viewHolder.mTopicImgProgressFavorite.setVisibility(0);
                    AnimUtil.rotate(viewHolder.mTopicImgProgressFavorite);
                    String str = ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i2)).getCollected().intValue() == 1 ? "uncollect" : "collect";
                    ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i2)).setCollectStatus(1);
                    CommunityAdapter communityAdapter2 = CommunityAdapter.this;
                    communityAdapter2.topicCollectAction(i2, ((TopicModel) communityAdapter2.mTopicModelList.get(i2)).getId(), str);
                }
            }
        });
        viewHolder.itemView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.adapter.CommunityAdapter.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                OnItemClickListener onItemClickListener = CommunityAdapter.this.onItemClickListener;
                CommunityAdapter communityAdapter2 = CommunityAdapter.this;
                if (onItemClickListener != null) {
                    communityAdapter2.onItemClickListener.onItemClick(i2);
                } else if (((TopicModel) communityAdapter2.mTopicModelList.get(i2)).getId() != null) {
                    Intent intent = new Intent(CommunityAdapter.this.mContext, (Class<?>) TopicContentActivity.class);
                    intent.putExtra("topicId", ((TopicModel) CommunityAdapter.this.mTopicModelList.get(i2)).getId());
                    CommunityAdapter.this.mContext.startActivity(intent);
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
            viewHolder.mTopicLayoutLike.setEnabled(true);
            viewHolder.mTopicImgProgressLike.clearAnimation();
            viewHolder.mTopicImgProgressLike.setVisibility(8);
            viewHolder.mTopicLayoutLike.setVisibility(0);
            viewHolder.mTopicTxtLike.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getLikeCount()));
            if (this.mTopicModelList.get(i).getLikeed().intValue() == 1) {
                viewHolder.mTopicImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
                return;
            } else {
                viewHolder.mTopicImgLike.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
                return;
            }
        }
        if ("collect".equals(list.get(0))) {
            viewHolder.mTopicLayoutFavorite.setEnabled(true);
            viewHolder.mTopicImgProgressFavorite.clearAnimation();
            viewHolder.mTopicImgProgressFavorite.setVisibility(8);
            viewHolder.mTopicLayoutFavorite.setVisibility(0);
            viewHolder.mTopicTxtFavorite.setText(StringUtils.getFormatNum(this.mTopicModelList.get(i).getCollectCount()));
            if (this.mTopicModelList.get(i).getCollected().intValue() == 1) {
                viewHolder.mTopicImgFavorite.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect_pressed));
            } else {
                viewHolder.mTopicImgFavorite.setImageDrawable(this.mContext.getResources().getDrawable(R.mipmap.ic_community_collect));
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int i) {
        LogUtil.e("空数据onCreateViewHolder：" + i);
        return i == 1 ? new ViewHolder(getEmptyView(viewGroup)) : new ViewHolder(LayoutInflater.from(this.mContext).inflate(R.layout.item_topic, viewGroup, false));
    }

    public void setOnItemClickListener(OnItemClickListener onItemClickListener) {
        this.onItemClickListener = onItemClickListener;
    }

    public void setOnUserInfoListener(OnUserInfoListener onUserInfoListener) {
        this.onUserInfoListener = onUserInfoListener;
    }

    public void setShowEmptyView(boolean z) {
        this.showEmptyView = z;
    }
}
