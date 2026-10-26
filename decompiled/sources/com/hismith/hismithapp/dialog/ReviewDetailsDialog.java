package com.hismith.hismithapp.dialog;

import android.animation.TypeEvaluator;
import android.animation.ValueAnimator;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.KeyEvent;
import android.view.View;
import android.view.animation.LinearInterpolator;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.core.widget.NestedScrollView;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.google.firebase.messaging.ServiceStarter;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.LikeUserActivity;
import com.hismith.hismithapp.PersonalDetailsActivity2;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.TopicContentActivity;
import com.hismith.hismithapp.UserDetailsActivity2;
import com.hismith.hismithapp.adapter.ReviewDetailsAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.HttpModel;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.listener.SoftKeyBoardListener;
import com.hismith.hismithapp.model.ReviewDetailsModel;
import com.hismith.hismithapp.model.ReviewDetailsModel.Reply;
import com.hismith.hismithapp.model.ReviewDetailsModel.Review;
import com.hismith.hismithapp.model.ReviewLikeModel;
import com.hismith.hismithapp.model.ReviewSendModel;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.utils.TimeUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.lxj.xpopup.XPopup;
import com.lxj.xpopup.core.BottomPopupView;
import com.lxj.xpopup.util.XPopupUtils;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnLoadMoreListener;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Objects;

/* loaded from: classes3.dex */
public class ReviewDetailsDialog extends BottomPopupView implements View.OnClickListener {
    private RetrofitRequest baseRequest;
    private boolean highlight;
    private ReviewDetailsAdapter mAdapter;
    private Context mContext;
    private EditText mEditReview;
    private Handler mHandler;
    private ImageView mImgDelete;
    private CircleImageView mImgHead;
    private ImageView mImgLike;
    private FrameLayout mLayoutHighlight;
    private ConstraintLayout mLayoutLike;
    private LinearLayout mLayoutLikeCount;
    private LinearLayoutManager mLayoutManager;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private ReviewDetailsModel mReviewDetailsModel;
    private List<ReviewDetailsModel.Review> mReviewList;
    private NestedScrollView mScrollView;
    private ImageView mTitleBack;
    private TextView mTitleName;
    private TextView mTxtContent;
    private TextView mTxtLikeCount;
    private TextView mTxtLikeCount1;
    private TextView mTxtName;
    private TextView mTxtPublish;
    private TextView mTxtReport;
    private TextView mTxtTime;
    private TextView mTxtTopicTitle;
    private int page;
    private int position;
    private int replyId;
    private Integer reviewId;
    private SimpleDateFormat sdf;
    private SoftKeyBoardListener softKeyBoardListener;
    private int totalPage;

    private class TextArgbEvaluator implements TypeEvaluator {
        private TextArgbEvaluator() {
        }

        @Override // android.animation.TypeEvaluator
        public Object evaluate(float f, Object obj, Object obj2) {
            int intValue = ((Integer) obj).intValue();
            int i = (intValue >> 24) & 255;
            int i2 = (intValue >> 16) & 255;
            int i3 = (intValue >> 8) & 255;
            int i4 = intValue & 255;
            int intValue2 = ((Integer) obj2).intValue();
            return Integer.valueOf((i4 + ((int) (f * ((intValue2 & 255) - i4)))) | ((i + ((int) ((((intValue2 >> 24) & 255) - i) * f))) << 24) | ((i2 + ((int) ((((intValue2 >> 16) & 255) - i2) * f))) << 16) | ((i3 + ((int) ((((intValue2 >> 8) & 255) - i3) * f))) << 8));
        }
    }

    public ReviewDetailsDialog(Context context) {
        super(context);
        this.replyId = -1;
        this.highlight = false;
        this.position = -1;
        this.page = 1;
        this.totalPage = 1;
        this.sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                return false;
            }
        });
        this.mContext = context;
    }

    public ReviewDetailsDialog(Context context, Integer num) {
        super(context);
        this.replyId = -1;
        this.highlight = false;
        this.position = -1;
        this.page = 1;
        this.totalPage = 1;
        this.sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                return false;
            }
        });
        this.mContext = context;
        this.reviewId = num;
    }

    public ReviewDetailsDialog(Context context, Integer num, int i, boolean z) {
        super(context);
        this.replyId = -1;
        this.highlight = false;
        this.position = -1;
        this.page = 1;
        this.totalPage = 1;
        this.sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                return false;
            }
        });
        this.mContext = context;
        this.reviewId = num;
        this.replyId = i;
        this.highlight = z;
    }

    public ReviewDetailsDialog(Context context, Integer num, boolean z) {
        super(context);
        this.replyId = -1;
        this.highlight = false;
        this.position = -1;
        this.page = 1;
        this.totalPage = 1;
        this.sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        this.mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.1
            @Override // android.os.Handler.Callback
            public boolean handleMessage(Message message) {
                return false;
            }
        });
        this.mContext = context;
        this.reviewId = num;
        this.highlight = z;
    }

    static /* synthetic */ int access$208(ReviewDetailsDialog reviewDetailsDialog) {
        int i = reviewDetailsDialog.page;
        reviewDetailsDialog.page = i + 1;
        return i;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void complaint(String str, String str2, String str3) {
        this.baseRequest.getApi().complaint((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.reviewId, str, str2, str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.9
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(ReviewDetailsDialog.this.mContext, LanguageUtil.getLanguageByKey(ReviewDetailsDialog.this.mContext, "reportedSuccess"));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getTopicReviewDetails() {
        Observable<ReviewDetailsModel> observeOn = this.baseRequest.getApi().topicReviewDetails((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.reviewId, Integer.valueOf(this.page)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        Context context = this.mContext;
        observeOn.subscribe(new RxSubscribe<ReviewDetailsModel>(context, ((BaseActivity) context).progressDialog) { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.5
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewDetailsModel reviewDetailsModel) {
                if (reviewDetailsModel.getSuccess() == 1) {
                    ReviewDetailsDialog.this.mReviewDetailsModel = reviewDetailsModel;
                    ReviewDetailsDialog.this.mTxtTopicTitle.setText(reviewDetailsModel.getTopicTitle());
                    Glide.with(ReviewDetailsDialog.this.mContext).load(reviewDetailsModel.getUserHeadIcon()).into((RequestBuilder<Drawable>) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.5.1
                        public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                            ReviewDetailsDialog.this.mImgHead.setImageDrawable(drawable);
                        }

                        @Override // com.bumptech.glide.request.target.Target
                        public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                            onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
                        }
                    });
                    ReviewDetailsDialog.this.mImgHead.setKing(reviewDetailsModel.getUserKing().intValue(), ReviewDetailsDialog.this.mContext, reviewDetailsModel.getGradeIcon());
                    ReviewDetailsDialog.this.mTxtName.setText(reviewDetailsModel.getUserName());
                    ReviewDetailsDialog.this.mTxtTime.setText(TimeUtil.utc2Local(reviewDetailsModel.getCreateDate()));
                    ReviewDetailsDialog.this.mTxtContent.setText(reviewDetailsModel.getContent());
                    ReviewDetailsDialog.this.mTxtLikeCount.setText(reviewDetailsModel.getLikeNote());
                    ReviewDetailsDialog.this.mTxtLikeCount1.setText(reviewDetailsModel.getLikeCount() + "");
                    int intValue = reviewDetailsModel.getLikeed().intValue();
                    ReviewDetailsDialog reviewDetailsDialog = ReviewDetailsDialog.this;
                    if (intValue == 1) {
                        reviewDetailsDialog.mImgLike.setImageDrawable(ReviewDetailsDialog.this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
                    } else {
                        reviewDetailsDialog.mImgLike.setImageDrawable(ReviewDetailsDialog.this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
                    }
                    if (reviewDetailsModel.getLikeUser() != null && reviewDetailsModel.getLikeUser().length > 0) {
                        for (int i = 0; i < reviewDetailsModel.getLikeUser().length && i < 3; i++) {
                            CircleImageView circleImageView = new CircleImageView(ReviewDetailsDialog.this.mContext);
                            circleImageView.setBackground(ReviewDetailsDialog.this.mContext.getResources().getDrawable(R.drawable.bg_circle_white));
                            Glide.with(ReviewDetailsDialog.this.mContext).load(reviewDetailsModel.getLikeUser()[i].getUserHeadIcon()).into(circleImageView);
                            circleImageView.setKing(reviewDetailsModel.getLikeUser()[i].getUserKing().intValue(), ReviewDetailsDialog.this.mContext, reviewDetailsModel.getLikeUser()[i].getGradeIcon());
                            LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(MeasureUtil.dip2px(ReviewDetailsDialog.this.mContext, 20.0f), MeasureUtil.dip2px(ReviewDetailsDialog.this.mContext, 20.0f));
                            layoutParams.setMarginStart(5);
                            layoutParams.setMarginEnd(5);
                            ReviewDetailsDialog.this.mLayoutLikeCount.addView(circleImageView, 0, layoutParams);
                        }
                    }
                    ReviewDetailsDialog.this.mReviewList = new ArrayList(Arrays.asList(reviewDetailsModel.getReplys()));
                    ReviewDetailsDialog.this.mAdapter = new ReviewDetailsAdapter(ReviewDetailsDialog.this.mContext, ReviewDetailsDialog.this.mReviewList);
                    ReviewDetailsDialog.this.mAdapter.setOnItemReplyClickListener(new ReviewDetailsAdapter.OnItemReplyClickListener() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.5.2
                        @Override // com.hismith.hismithapp.adapter.ReviewDetailsAdapter.OnItemReplyClickListener
                        public void itemReplyClick(int i2) {
                            LogUtil.e("回复评论position:" + i2 + ", id:" + ((ReviewDetailsModel.Review) ReviewDetailsDialog.this.mReviewList.get(i2)).getId() + ", content:" + ((ReviewDetailsModel.Review) ReviewDetailsDialog.this.mReviewList.get(i2)).getContent());
                            ReviewDetailsDialog.this.position = i2;
                            ReviewDetailsDialog.this.mEditReview.setHint(LanguageUtil.getLanguageByKey(ReviewDetailsDialog.this.mContext, "reply") + " " + (((ReviewDetailsModel.Review) ReviewDetailsDialog.this.mReviewList.get(i2)).getUserName() != null ? ((ReviewDetailsModel.Review) ReviewDetailsDialog.this.mReviewList.get(i2)).getUserName() : "") + ":");
                            SoftKeyboardUtil.showInput(ReviewDetailsDialog.this.mContext, ReviewDetailsDialog.this.mEditReview);
                        }
                    });
                    ReviewDetailsDialog.this.mAdapter.setReviewDelClickListener(new ReviewDetailsAdapter.ReviewDelClickListener() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.5.3
                        @Override // com.hismith.hismithapp.adapter.ReviewDetailsAdapter.ReviewDelClickListener
                        public void onReviewDelClick(int i2) {
                            ReviewDetailsDialog.this.topicReviewDel(Integer.valueOf(i2));
                        }
                    });
                    ReviewDetailsDialog.this.mRecyclerView.setAdapter(ReviewDetailsDialog.this.mAdapter);
                    ReviewDetailsDialog.this.mHandler.postDelayed(new Runnable() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.5.4
                        @Override // java.lang.Runnable
                        public void run() {
                            if (ReviewDetailsDialog.this.highlight) {
                                if (ReviewDetailsDialog.this.replyId == -1) {
                                    ReviewDetailsDialog.this.initAnimator(ReviewDetailsDialog.this.mLayoutHighlight);
                                    return;
                                }
                                for (int i2 = 0; i2 < ReviewDetailsDialog.this.mReviewList.size(); i2++) {
                                    LogUtil.e("replyId:" + ReviewDetailsDialog.this.replyId + "=?" + ((ReviewDetailsModel.Review) ReviewDetailsDialog.this.mReviewList.get(i2)).getId());
                                    if (((ReviewDetailsModel.Review) ReviewDetailsDialog.this.mReviewList.get(i2)).getId().intValue() == ReviewDetailsDialog.this.replyId) {
                                        RecyclerView.ViewHolder findViewHolderForAdapterPosition = ReviewDetailsDialog.this.mRecyclerView.findViewHolderForAdapterPosition(i2);
                                        if (findViewHolderForAdapterPosition != null) {
                                            View view = findViewHolderForAdapterPosition.itemView;
                                            LogUtil.e("smoothScrollTo top:" + view.getTop());
                                            ReviewDetailsDialog.this.mScrollView.smoothScrollTo(0, view.getTop());
                                            ReviewDetailsDialog.this.initAnimator(view);
                                            return;
                                        }
                                        return;
                                    }
                                }
                            }
                        }
                    }, 500L);
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void initAnimator(final View view) {
        ValueAnimator valueAnimator = new ValueAnimator();
        valueAnimator.setIntValues(getResources().getColor(R.color.app_style_color), Color.parseColor("#00322054"));
        valueAnimator.setEvaluator(new TextArgbEvaluator());
        valueAnimator.addUpdateListener(new ValueAnimator.AnimatorUpdateListener() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.2
            @Override // android.animation.ValueAnimator.AnimatorUpdateListener
            public void onAnimationUpdate(ValueAnimator valueAnimator2) {
                int intValue = ((Integer) valueAnimator2.getAnimatedValue()).intValue();
                LogUtil.e("color:" + intValue);
                view.setBackgroundColor(intValue);
            }
        });
        valueAnimator.setInterpolator(new LinearInterpolator());
        valueAnimator.setDuration(800L);
        valueAnimator.setRepeatCount(0);
        valueAnimator.start();
    }

    private void initData() {
        this.baseRequest = new RetrofitRequest();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.mContext);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mRefreshLayout.setEnableRefresh(false);
        this.mRefreshLayout.setEnableLoadMore(false);
        getTopicReviewDetails();
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.mImgHead = (CircleImageView) findViewById(R.id.review_details_img_head);
        this.mTxtName = (TextView) findViewById(R.id.review_details_txt_name);
        this.mTxtTime = (TextView) findViewById(R.id.review_details_txt_time);
        this.mTxtContent = (TextView) findViewById(R.id.review_details_txt_content);
        this.mTxtReport = (TextView) findViewById(R.id.review_details_txt_report);
        this.mTxtLikeCount = (TextView) findViewById(R.id.review_details_txt_likeCount);
        this.mTxtLikeCount1 = (TextView) findViewById(R.id.review_details_txt_likeCount_1);
        this.mImgDelete = (ImageView) findViewById(R.id.review_details_img_delete);
        this.mImgLike = (ImageView) findViewById(R.id.review_details_img_like);
        this.mLayoutLike = (ConstraintLayout) findViewById(R.id.review_details_layout_like);
        this.mRefreshLayout = (SmartRefreshLayout) findViewById(R.id.review_details_refreshLayout);
        this.mScrollView = (NestedScrollView) findViewById(R.id.scroll_view);
        this.mRecyclerView = (RecyclerView) findViewById(R.id.review_details_recycler);
        this.mEditReview = (EditText) findViewById(R.id.review_details_edit_review);
        this.mTxtPublish = (TextView) findViewById(R.id.review_details_txt_publish);
        this.mLayoutLikeCount = (LinearLayout) findViewById(R.id.review_details_layout_likeCount);
        this.mLayoutHighlight = (FrameLayout) findViewById(R.id.layout_highlight);
        this.mTxtTopicTitle = (TextView) findViewById(R.id.review_details_txt_topic_title);
        LanguageUtil.setText(this.mContext, this.mTxtReport, "report");
        LanguageUtil.setText(this.mContext, this.mTxtPublish, "publish");
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void sendReview(View view) {
        int i = this.position;
        if (i != -1) {
            topicReviewSend(i, this.mReviewList.get(i).getId().intValue());
        } else {
            topicReviewSend(this.reviewId.intValue());
        }
        SoftKeyboardUtil.hideSoftKeyboard(this.mContext, view);
    }

    private void setListener() {
        setSoftKeyBoardListener();
        this.mTitleBack.setOnClickListener(this);
        this.mImgHead.setOnClickListener(this);
        this.mTxtName.setOnClickListener(this);
        this.mImgLike.setOnClickListener(this);
        this.mTxtReport.setOnClickListener(this);
        this.mLayoutLikeCount.setOnClickListener(this);
        this.mTxtTopicTitle.setOnClickListener(this);
        this.mTxtPublish.setOnClickListener(this);
        this.mEditReview.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.3
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 4) {
                    return false;
                }
                ReviewDetailsDialog.this.sendReview(textView);
                return false;
            }
        });
        this.mRefreshLayout.setOnLoadMoreListener(new OnLoadMoreListener() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.4
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = ReviewDetailsDialog.this.page;
                int i2 = ReviewDetailsDialog.this.totalPage;
                ReviewDetailsDialog reviewDetailsDialog = ReviewDetailsDialog.this;
                if (i >= i2) {
                    reviewDetailsDialog.mRefreshLayout.finishLoadMore(ServiceStarter.ERROR_UNKNOWN);
                } else {
                    ReviewDetailsDialog.access$208(reviewDetailsDialog);
                    ReviewDetailsDialog.this.getTopicReviewDetails();
                }
            }
        });
    }

    private void setSoftKeyBoardListener() {
        SoftKeyBoardListener softKeyBoardListener = new SoftKeyBoardListener((Activity) this.mContext);
        this.softKeyBoardListener = softKeyBoardListener;
        softKeyBoardListener.setListener(new SoftKeyBoardListener.OnSoftKeyBoardChangeListener() { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.12
            @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
            public void keyBoardHide(int i) {
                LogUtil.e("隐藏软键盘");
                ReviewDetailsDialog.this.mEditReview.setHint("");
                ReviewDetailsDialog.this.position = -1;
            }

            @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
            public void keyBoardShow(int i) {
                LogUtil.e("显示软键盘");
            }
        });
    }

    private void topicReviewDel() {
        if (this.mReviewDetailsModel == null) {
            return;
        }
        Observable<HttpModel> observeOn = this.baseRequest.getApi().topicReviewDel((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mReviewDetailsModel.getId()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        Context context = this.mContext;
        observeOn.subscribe(new RxSubscribe<HttpModel>(context, ((BaseActivity) context).progressDialog, true) { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.6
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ReviewDetailsDialog.this.dismiss();
                    ToastUtil.showShortToastCenter(ReviewDetailsDialog.this.mContext, LanguageUtil.getLanguageByKey(ReviewDetailsDialog.this.mContext, "deletedMsg"));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicReviewDel(final Integer num) {
        Observable<HttpModel> observeOn = this.baseRequest.getApi().topicReviewDel((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), this.mReviewList.get(num.intValue()).getId()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        Context context = this.mContext;
        observeOn.subscribe(new RxSubscribe<HttpModel>(context, ((BaseActivity) context).progressDialog) { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.10
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ReviewDetailsDialog.this.mReviewList.remove(num.intValue());
                    ToastUtil.showShortToastCenter(ReviewDetailsDialog.this.mContext, LanguageUtil.getLanguageByKey(ReviewDetailsDialog.this.mContext, "deletedMsg"));
                    ReviewDetailsDialog.this.mAdapter.notifyItemRemoved(num.intValue());
                    ReviewDetailsDialog.this.mAdapter.notifyItemRangeChanged(num.intValue(), ReviewDetailsDialog.this.mReviewList.size());
                }
            }
        });
    }

    private void topicReviewLike(Integer num, final String str) {
        this.baseRequest.getApi().topicReviewLike((String) SharedPreferencesUtil.get(this.mContext, Constans.Token), num, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ReviewLikeModel>(this.mContext, null, false) { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.8
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewLikeModel reviewLikeModel) {
                if (reviewLikeModel.getSuccess() == 1) {
                    if ("unlike".equals(str)) {
                        ReviewDetailsDialog.this.mReviewDetailsModel.setLikeed(0);
                    } else if ("like".equals(str)) {
                        ReviewDetailsDialog.this.mReviewDetailsModel.setLikeed(1);
                    }
                    ReviewDetailsDialog.this.mReviewDetailsModel.setLikeCount(reviewLikeModel.getLikeCount());
                    ReviewDetailsDialog.this.mTxtLikeCount1.setText(ReviewDetailsDialog.this.mReviewDetailsModel.getLikeCount() + "");
                    int intValue = ReviewDetailsDialog.this.mReviewDetailsModel.getLikeed().intValue();
                    ReviewDetailsDialog reviewDetailsDialog = ReviewDetailsDialog.this;
                    if (intValue == 1) {
                        reviewDetailsDialog.mImgLike.setImageDrawable(ReviewDetailsDialog.this.mContext.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
                    } else {
                        reviewDetailsDialog.mImgLike.setImageDrawable(ReviewDetailsDialog.this.mContext.getResources().getDrawable(R.mipmap.ic_community_like));
                    }
                }
            }
        });
    }

    private void topicReviewSend(int i) {
        topicReviewSend(-1, i);
    }

    private void topicReviewSend(final int i, int i2) {
        String str = (String) SharedPreferencesUtil.get(this.mContext, Constans.Token);
        final String obj = this.mEditReview.getText().toString();
        if (this.mReviewDetailsModel == null || TextUtils.isEmpty(obj)) {
            return;
        }
        Observable<ReviewSendModel> observeOn = this.baseRequest.getApi().topicReviewReplySend(str, this.mReviewDetailsModel.getTopicId(), obj, Integer.valueOf(i2)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        Context context = this.mContext;
        observeOn.subscribe(new RxSubscribe<ReviewSendModel>(context, ((BaseActivity) context).progressDialog, true) { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.7
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewSendModel reviewSendModel) {
                if (reviewSendModel.getSuccess() == 1) {
                    ReviewDetailsDialog.this.mEditReview.setText("");
                    ReviewDetailsModel reviewDetailsModel = new ReviewDetailsModel();
                    Objects.requireNonNull(reviewDetailsModel);
                    ReviewDetailsModel.Review review = reviewDetailsModel.new Review();
                    review.setId(reviewSendModel.getRid());
                    review.setLikeed(0);
                    review.setLikeCount(0);
                    review.setContent(obj);
                    review.setCreateDate(ReviewDetailsDialog.this.sdf.format(new Date()));
                    UserModel userModel = MyApplication.getApplication().getmUserModel();
                    if (userModel != null) {
                        review.setUserGender(Integer.valueOf(userModel.getGender()));
                        review.setUserHeadIcon(userModel.getHeadIcon());
                        review.setUserId(Integer.valueOf(userModel.getUserId()));
                        review.setUserKing(Integer.valueOf(userModel.getKing()));
                        review.setUserName(userModel.getName());
                    }
                    if (i != -1) {
                        Objects.requireNonNull(reviewDetailsModel);
                        ReviewDetailsModel.Reply reply = reviewDetailsModel.new Reply();
                        reply.setContent(((ReviewDetailsModel.Review) ReviewDetailsDialog.this.mReviewList.get(i)).getContent());
                        reply.setId(((ReviewDetailsModel.Review) ReviewDetailsDialog.this.mReviewList.get(i)).getId());
                        if (userModel != null) {
                            reply.setUserGender(Integer.valueOf(userModel.getGender()));
                            reply.setUserHeadIcon(userModel.getHeadIcon());
                            reply.setUserId(Integer.valueOf(userModel.getUserId()));
                            reply.setUserKing(Integer.valueOf(userModel.getKing()));
                            reply.setUserName(userModel.getName());
                        }
                        review.setReply(reply);
                    }
                    ReviewDetailsDialog.this.mReviewList.add(0, review);
                    ReviewDetailsDialog.this.mAdapter.notifyDataSetChanged();
                    ReviewDetailsDialog.this.mRecyclerView.scrollToPosition(ReviewDetailsDialog.this.mReviewList.size());
                }
            }
        });
    }

    @Override // com.lxj.xpopup.core.BottomPopupView, com.lxj.xpopup.core.BasePopupView
    protected int getImplLayoutId() {
        return R.layout.activity_review_details;
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected int getPopupHeight() {
        return (int) (XPopupUtils.getScreenHeight(getContext()) * 0.9f);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        ReviewDetailsModel reviewDetailsModel;
        ReviewDetailsModel reviewDetailsModel2;
        int id = view.getId();
        if (id == R.id.title_img_back) {
            dismiss();
            return;
        }
        if (id == R.id.review_details_img_head || id == R.id.review_details_txt_name) {
            if (MyApplication.getApplication().getmUserModel() == null || (reviewDetailsModel = this.mReviewDetailsModel) == null) {
                return;
            }
            if (reviewDetailsModel.getUserId() == MyApplication.getApplication().getmUserModel().getUserId()) {
                this.mContext.startActivity(new Intent(this.mContext, (Class<?>) PersonalDetailsActivity2.class));
                return;
            }
            Intent intent = new Intent(this.mContext, (Class<?>) UserDetailsActivity2.class);
            intent.putExtra("userId", this.mReviewDetailsModel.getUserId());
            this.mContext.startActivity(intent);
            return;
        }
        if (id == R.id.review_details_img_like) {
            ReviewDetailsModel reviewDetailsModel3 = this.mReviewDetailsModel;
            if (reviewDetailsModel3 != null) {
                int intValue = reviewDetailsModel3.getLikeed().intValue();
                ReviewDetailsModel reviewDetailsModel4 = this.mReviewDetailsModel;
                if (intValue == 1) {
                    topicReviewLike(reviewDetailsModel4.getId(), "unlike");
                    return;
                } else {
                    topicReviewLike(reviewDetailsModel4.getId(), "like");
                    return;
                }
            }
            return;
        }
        if (id == R.id.review_details_txt_report) {
            if (this.mReviewDetailsModel != null) {
                new ReportSelectDialog(this.mContext) { // from class: com.hismith.hismithapp.dialog.ReviewDetailsDialog.11
                    @Override // com.hismith.hismithapp.dialog.ReportSelectDialog
                    public void btnSubmit(String str, String str2) {
                        ReviewDetailsDialog.this.complaint("review", str, str2);
                    }
                }.showDialog();
                return;
            }
            return;
        }
        if (id == R.id.review_details_layout_likeCount) {
            if (this.mReviewDetailsModel != null) {
                Intent intent2 = new Intent(this.mContext, (Class<?>) LikeUserActivity.class);
                intent2.putExtra("reviewId", this.reviewId);
                this.mContext.startActivity(intent2);
                return;
            }
            return;
        }
        if (id == R.id.review_details_txt_publish) {
            sendReview(view);
            return;
        }
        if (id == R.id.review_details_img_delete) {
            topicReviewDel();
            return;
        }
        if (id != R.id.review_details_txt_topic_title || (reviewDetailsModel2 = this.mReviewDetailsModel) == null || reviewDetailsModel2.getTopicId().intValue() == 0) {
            return;
        }
        Intent intent3 = new Intent(this.mContext, (Class<?>) TopicContentActivity.class);
        intent3.putExtra("topicId", this.mReviewDetailsModel.getTopicId());
        this.mContext.startActivity(intent3);
    }

    @Override // com.lxj.xpopup.core.BasePopupView
    protected void onCreate() {
        super.onCreate();
        initView();
        initData();
        setListener();
    }

    public void showDialog() {
        new XPopup.Builder(this.mContext).autoFocusEditText(false).autoOpenSoftInput(false).asCustom(this).show();
    }
}
