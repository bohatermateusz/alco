package com.hismith.hismithapp;

import android.content.Context;
import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.TypedValue;
import android.view.KeyEvent;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.fragment.app.FragmentActivity;
import androidx.recyclerview.widget.DividerItemDecoration;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.google.firebase.messaging.ServiceStarter;
import com.hismith.hismithapp.adapter.ReviewDetailsAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.dialog.ProgressDialog;
import com.hismith.hismithapp.dialog.ReportSelectDialog;
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
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.utils.TimeUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnLoadMoreListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Objects;

/* loaded from: classes3.dex */
public class ReviewDetailsActivity extends BaseActivity implements View.OnClickListener, ReviewDetailsAdapter.OnItemReplyClickListener, ReviewDetailsAdapter.ReviewDelClickListener {
    private RetrofitRequest baseRequest;
    private ReviewDetailsAdapter mAdapter;
    private EditText mEditReview;
    private ImageView mImgDelete;
    private CircleImageView mImgHead;
    private ImageView mImgLike;
    private LinearLayout mLayoutLike;
    private LinearLayout mLayoutLikeCount;
    private LinearLayoutManager mLayoutManager;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private ReviewDetailsModel mReviewDetailsModel;
    private List<ReviewDetailsModel.Review> mReviewList;
    private ImageView mTitleBack;
    private TextView mTitleName;
    private TextView mTxtContent;
    private TextView mTxtLikeCount;
    private TextView mTxtLikeCount1;
    private TextView mTxtName;
    private TextView mTxtPublish;
    private TextView mTxtReport;
    private TextView mTxtTime;
    private Integer reviewId;
    private SoftKeyBoardListener softKeyBoardListener;
    private int position = -1;
    private int page = 1;
    private int totalPage = 1;
    private SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

    /* renamed from: com.hismith.hismithapp.ReviewDetailsActivity$3, reason: invalid class name */
    class AnonymousClass3 extends RxSubscribe<ReviewDetailsModel> {
        AnonymousClass3(Context context, ProgressDialog progressDialog) {
            super(context, progressDialog);
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.hismith.hismithapp.http.RxSubscribe
        public void onSuccess(ReviewDetailsModel reviewDetailsModel) {
            if (reviewDetailsModel.getSuccess() == 1) {
                ReviewDetailsActivity.this.mReviewDetailsModel = reviewDetailsModel;
                Glide.with((FragmentActivity) ReviewDetailsActivity.this).load(reviewDetailsModel.getUserHeadIcon()).into((RequestBuilder<Drawable>) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.ReviewDetailsActivity.3.1
                    public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                        ReviewDetailsActivity.this.mImgHead.setImageDrawable(drawable);
                    }

                    @Override // com.bumptech.glide.request.target.Target
                    public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                        onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
                    }
                });
                ReviewDetailsActivity.this.mImgHead.setKing(reviewDetailsModel.getUserKing().intValue(), ReviewDetailsActivity.this, reviewDetailsModel.getGradeIcon());
                ReviewDetailsActivity.this.mTxtName.setText(reviewDetailsModel.getUserName());
                ReviewDetailsActivity.this.mTxtTime.setText(TimeUtil.utc2Local(reviewDetailsModel.getCreateDate()));
                ReviewDetailsActivity.this.mTxtContent.setText(reviewDetailsModel.getContent());
                ReviewDetailsActivity.this.mTxtLikeCount.setText(reviewDetailsModel.getLikeNote());
                ReviewDetailsActivity.this.mTxtLikeCount1.setText(reviewDetailsModel.getLikeCount() + "");
                int intValue = reviewDetailsModel.getLikeed().intValue();
                ReviewDetailsActivity reviewDetailsActivity = ReviewDetailsActivity.this;
                if (intValue == 1) {
                    reviewDetailsActivity.mImgLike.setImageDrawable(ReviewDetailsActivity.this.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
                } else {
                    reviewDetailsActivity.mImgLike.setImageDrawable(ReviewDetailsActivity.this.getResources().getDrawable(R.mipmap.ic_community_like));
                }
                if (reviewDetailsModel.getLikeUser() != null && reviewDetailsModel.getLikeUser().length > 0) {
                    for (int i = 0; i < reviewDetailsModel.getLikeUser().length && i < 3; i++) {
                        CircleImageView circleImageView = new CircleImageView(ReviewDetailsActivity.this);
                        circleImageView.setBackground(ReviewDetailsActivity.this.getResources().getDrawable(R.drawable.bg_circle_white));
                        Glide.with((FragmentActivity) ReviewDetailsActivity.this).load(reviewDetailsModel.getLikeUser()[i].getUserHeadIcon()).into(circleImageView);
                        circleImageView.setKing(reviewDetailsModel.getLikeUser()[i].getUserKing().intValue(), ReviewDetailsActivity.this, reviewDetailsModel.getLikeUser()[i].getGradeIcon());
                        LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(ReviewDetailsActivity.this.dp2px(20), ReviewDetailsActivity.this.dp2px(20));
                        layoutParams.setMarginStart(5);
                        layoutParams.setMarginEnd(5);
                        ReviewDetailsActivity.this.mLayoutLikeCount.addView(circleImageView, 0, layoutParams);
                    }
                }
                ReviewDetailsActivity.this.mReviewList = new ArrayList(Arrays.asList(reviewDetailsModel.getReplys()));
                ReviewDetailsActivity reviewDetailsActivity2 = ReviewDetailsActivity.this;
                ReviewDetailsActivity reviewDetailsActivity3 = ReviewDetailsActivity.this;
                reviewDetailsActivity2.mAdapter = new ReviewDetailsAdapter(reviewDetailsActivity3, reviewDetailsActivity3.mReviewList);
                ReviewDetailsAdapter reviewDetailsAdapter = ReviewDetailsActivity.this.mAdapter;
                final ReviewDetailsActivity reviewDetailsActivity4 = ReviewDetailsActivity.this;
                reviewDetailsAdapter.setOnItemReplyClickListener(new ReviewDetailsAdapter.OnItemReplyClickListener() { // from class: com.hismith.hismithapp.ReviewDetailsActivity$3$$ExternalSyntheticLambda0
                    @Override // com.hismith.hismithapp.adapter.ReviewDetailsAdapter.OnItemReplyClickListener
                    public final void itemReplyClick(int i2) {
                        ReviewDetailsActivity.this.itemReplyClick(i2);
                    }
                });
                ReviewDetailsAdapter reviewDetailsAdapter2 = ReviewDetailsActivity.this.mAdapter;
                final ReviewDetailsActivity reviewDetailsActivity5 = ReviewDetailsActivity.this;
                reviewDetailsAdapter2.setReviewDelClickListener(new ReviewDetailsAdapter.ReviewDelClickListener() { // from class: com.hismith.hismithapp.ReviewDetailsActivity$3$$ExternalSyntheticLambda1
                    @Override // com.hismith.hismithapp.adapter.ReviewDetailsAdapter.ReviewDelClickListener
                    public final void onReviewDelClick(int i2) {
                        ReviewDetailsActivity.this.topicReviewDel(Integer.valueOf(i2));
                    }
                });
                ReviewDetailsActivity.this.mRecyclerView.setAdapter(ReviewDetailsActivity.this.mAdapter);
            }
        }
    }

    static /* synthetic */ int access$508(ReviewDetailsActivity reviewDetailsActivity) {
        int i = reviewDetailsActivity.page;
        reviewDetailsActivity.page = i + 1;
        return i;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void complaint(String str, String str2, String str3) {
        this.baseRequest.getApi().complaint((String) SharedPreferencesUtil.get(this, Constans.Token), this.reviewId, str, str2, str3).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, null, false) { // from class: com.hismith.hismithapp.ReviewDetailsActivity.6
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ToastUtil.showShortToastCenter(ReviewDetailsActivity.this, LanguageUtil.getLanguageByKey(ReviewDetailsActivity.this, "reportedSuccess"));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getTopicReviewDetails() {
        this.baseRequest.getApi().topicReviewDetails((String) SharedPreferencesUtil.get(this, Constans.Token), this.reviewId, Integer.valueOf(this.page)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new AnonymousClass3(this, this.progressDialog));
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.reviewId = Integer.valueOf(getIntent().getIntExtra("reviewId", 0));
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mRecyclerView.addItemDecoration(new DividerItemDecoration(this, 1));
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
        this.mLayoutLike = (LinearLayout) findViewById(R.id.review_details_layout_like);
        this.mRefreshLayout = (SmartRefreshLayout) findViewById(R.id.review_details_refreshLayout);
        this.mRecyclerView = (RecyclerView) findViewById(R.id.review_details_recycler);
        this.mEditReview = (EditText) findViewById(R.id.review_details_edit_review);
        this.mTxtPublish = (TextView) findViewById(R.id.review_details_txt_publish);
        this.mLayoutLikeCount = (LinearLayout) findViewById(R.id.review_details_layout_likeCount);
        LanguageUtil.setText(this, this.mTxtReport, "report");
    }

    private void setListener() {
        setSoftKeyBoardListener();
        this.mTitleBack.setOnClickListener(this);
        this.mImgHead.setOnClickListener(this);
        this.mTxtName.setOnClickListener(this);
        this.mImgLike.setOnClickListener(this);
        this.mTxtReport.setOnClickListener(this);
        this.mLayoutLikeCount.setOnClickListener(this);
        this.mEditReview.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.ReviewDetailsActivity.1
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 4) {
                    return false;
                }
                int i2 = ReviewDetailsActivity.this.position;
                ReviewDetailsActivity reviewDetailsActivity = ReviewDetailsActivity.this;
                if (i2 != -1) {
                    reviewDetailsActivity.topicReviewSend(reviewDetailsActivity.position, ((ReviewDetailsModel.Review) ReviewDetailsActivity.this.mReviewList.get(ReviewDetailsActivity.this.position)).getId().intValue());
                } else {
                    reviewDetailsActivity.topicReviewSend(reviewDetailsActivity.reviewId.intValue());
                }
                SoftKeyboardUtil.hideSoftKeyboard(ReviewDetailsActivity.this, textView);
                return false;
            }
        });
        this.mRefreshLayout.setOnLoadMoreListener(new OnLoadMoreListener() { // from class: com.hismith.hismithapp.ReviewDetailsActivity.2
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = ReviewDetailsActivity.this.page;
                int i2 = ReviewDetailsActivity.this.totalPage;
                ReviewDetailsActivity reviewDetailsActivity = ReviewDetailsActivity.this;
                if (i >= i2) {
                    reviewDetailsActivity.mRefreshLayout.finishLoadMore(ServiceStarter.ERROR_UNKNOWN);
                } else {
                    ReviewDetailsActivity.access$508(reviewDetailsActivity);
                    ReviewDetailsActivity.this.getTopicReviewDetails();
                }
            }
        });
    }

    private void setSoftKeyBoardListener() {
        SoftKeyBoardListener softKeyBoardListener = new SoftKeyBoardListener(this);
        this.softKeyBoardListener = softKeyBoardListener;
        softKeyBoardListener.setListener(new SoftKeyBoardListener.OnSoftKeyBoardChangeListener() { // from class: com.hismith.hismithapp.ReviewDetailsActivity.9
            @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
            public void keyBoardHide(int i) {
                LogUtil.e("隐藏软键盘");
                ReviewDetailsActivity.this.mEditReview.setHint("");
                ReviewDetailsActivity.this.position = -1;
            }

            @Override // com.hismith.hismithapp.listener.SoftKeyBoardListener.OnSoftKeyBoardChangeListener
            public void keyBoardShow(int i) {
                LogUtil.e("显示软键盘");
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicReviewDel(final Integer num) {
        this.baseRequest.getApi().topicReviewDel((String) SharedPreferencesUtil.get(this, Constans.Token), this.mReviewList.get(num.intValue()).getId()).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<HttpModel>(this, this.progressDialog, true) { // from class: com.hismith.hismithapp.ReviewDetailsActivity.7
            @Override // com.hismith.hismithapp.http.RxSubscribe
            protected void onSuccess(HttpModel httpModel) {
                if (httpModel.getSuccess() == 1) {
                    ReviewDetailsActivity.this.mReviewList.remove(num.intValue());
                    ToastUtil.showShortToastCenter(ReviewDetailsActivity.this, LanguageUtil.getLanguageByKey(ReviewDetailsActivity.this, "deletedMsg"));
                    ReviewDetailsActivity.this.mAdapter.notifyItemRemoved(num.intValue());
                    ReviewDetailsActivity.this.mAdapter.notifyItemRangeChanged(num.intValue(), ReviewDetailsActivity.this.mReviewList.size());
                }
            }
        });
    }

    private void topicReviewLike(Integer num, final String str) {
        this.baseRequest.getApi().topicReviewLike((String) SharedPreferencesUtil.get(this, Constans.Token), num, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ReviewLikeModel>(this, null, false) { // from class: com.hismith.hismithapp.ReviewDetailsActivity.5
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewLikeModel reviewLikeModel) {
                if (reviewLikeModel.getSuccess() == 1) {
                    if ("unlike".equals(str)) {
                        ReviewDetailsActivity.this.mReviewDetailsModel.setLikeed(0);
                    } else if ("like".equals(str)) {
                        ReviewDetailsActivity.this.mReviewDetailsModel.setLikeed(1);
                    }
                    ReviewDetailsActivity.this.mReviewDetailsModel.setLikeCount(reviewLikeModel.getLikeCount());
                    ReviewDetailsActivity.this.mTxtLikeCount.setText(ReviewDetailsActivity.this.mReviewDetailsModel.getLikeCount() + "");
                    ReviewDetailsActivity.this.mTxtLikeCount1.setText(ReviewDetailsActivity.this.mReviewDetailsModel.getLikeCount() + "");
                    int intValue = ReviewDetailsActivity.this.mReviewDetailsModel.getLikeed().intValue();
                    ReviewDetailsActivity reviewDetailsActivity = ReviewDetailsActivity.this;
                    if (intValue == 1) {
                        reviewDetailsActivity.mImgLike.setImageDrawable(ReviewDetailsActivity.this.getResources().getDrawable(R.mipmap.ic_community_like_pressed));
                    } else {
                        reviewDetailsActivity.mImgLike.setImageDrawable(ReviewDetailsActivity.this.getResources().getDrawable(R.mipmap.ic_community_like));
                    }
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicReviewSend(int i) {
        topicReviewSend(-1, i);
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicReviewSend(final int i, int i2) {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        final String obj = this.mEditReview.getText().toString();
        if (this.mReviewDetailsModel == null || TextUtils.isEmpty(obj)) {
            return;
        }
        this.baseRequest.getApi().topicReviewReplySend(str, this.mReviewDetailsModel.getTopicId(), obj, Integer.valueOf(i2)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ReviewSendModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.ReviewDetailsActivity.4
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewSendModel reviewSendModel) {
                if (reviewSendModel.getSuccess() == 1) {
                    ReviewDetailsActivity.this.mEditReview.setText("");
                    ReviewDetailsModel reviewDetailsModel = new ReviewDetailsModel();
                    Objects.requireNonNull(reviewDetailsModel);
                    ReviewDetailsModel.Review review = reviewDetailsModel.new Review();
                    review.setId(reviewSendModel.getRid());
                    review.setLikeed(0);
                    review.setLikeCount(0);
                    review.setContent(obj);
                    review.setCreateDate(ReviewDetailsActivity.this.sdf.format(new Date()));
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
                        reply.setContent(((ReviewDetailsModel.Review) ReviewDetailsActivity.this.mReviewList.get(i)).getContent());
                        reply.setId(((ReviewDetailsModel.Review) ReviewDetailsActivity.this.mReviewList.get(i)).getId());
                        if (userModel != null) {
                            reply.setUserGender(Integer.valueOf(userModel.getGender()));
                            reply.setUserHeadIcon(userModel.getHeadIcon());
                            reply.setUserId(Integer.valueOf(userModel.getUserId()));
                            reply.setUserKing(Integer.valueOf(userModel.getKing()));
                            reply.setUserName(userModel.getName());
                        }
                        review.setReply(reply);
                    }
                    ReviewDetailsActivity.this.mReviewList.add(0, review);
                    ReviewDetailsActivity.this.mAdapter.notifyDataSetChanged();
                    ReviewDetailsActivity.this.mRecyclerView.scrollToPosition(ReviewDetailsActivity.this.mReviewList.size());
                }
            }
        });
    }

    protected int dp2px(int i) {
        return (int) TypedValue.applyDimension(1, i, getResources().getDisplayMetrics());
    }

    @Override // com.hismith.hismithapp.adapter.ReviewDetailsAdapter.OnItemReplyClickListener
    public void itemReplyClick(int i) {
        LogUtil.e("回复评论position:" + i + ", id:" + this.mReviewList.get(i).getId() + ", content:" + this.mReviewList.get(i).getContent());
        this.position = i;
        this.mEditReview.setHint(LanguageUtil.getLanguageByKey(this, "reply") + " " + (this.mReviewList.get(i).getUserName() != null ? this.mReviewList.get(i).getUserName() : "") + ":");
        SoftKeyboardUtil.showInput(this, this.mEditReview);
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        ReviewDetailsModel reviewDetailsModel;
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.review_details_img_head || id == R.id.review_details_txt_name) {
            if (MyApplication.getApplication().getmUserModel() == null || (reviewDetailsModel = this.mReviewDetailsModel) == null) {
                return;
            }
            if (reviewDetailsModel.getUserId() == MyApplication.getApplication().getmUserModel().getUserId()) {
                startActivity(new Intent(this, (Class<?>) PersonalDetailsActivity2.class));
                return;
            }
            Intent intent = new Intent(this, (Class<?>) UserDetailsActivity2.class);
            intent.putExtra("userId", this.mReviewDetailsModel.getUserId());
            startActivity(intent);
            return;
        }
        if (id == R.id.review_details_img_like) {
            ReviewDetailsModel reviewDetailsModel2 = this.mReviewDetailsModel;
            if (reviewDetailsModel2 != null) {
                int intValue = reviewDetailsModel2.getLikeed().intValue();
                ReviewDetailsModel reviewDetailsModel3 = this.mReviewDetailsModel;
                if (intValue == 1) {
                    topicReviewLike(reviewDetailsModel3.getId(), "unlike");
                    return;
                } else {
                    topicReviewLike(reviewDetailsModel3.getId(), "like");
                    return;
                }
            }
            return;
        }
        if (id == R.id.review_details_txt_report) {
            if (this.mReviewDetailsModel != null) {
                new ReportSelectDialog(this) { // from class: com.hismith.hismithapp.ReviewDetailsActivity.8
                    @Override // com.hismith.hismithapp.dialog.ReportSelectDialog
                    public void btnSubmit(String str, String str2) {
                        ReviewDetailsActivity.this.complaint("review", str, str2);
                    }
                }.showDialog();
            }
        } else {
            if (id != R.id.review_details_layout_likeCount || this.mReviewDetailsModel == null) {
                return;
            }
            Intent intent2 = new Intent(this, (Class<?>) LikeUserActivity.class);
            intent2.putExtra("reviewId", this.reviewId);
            startActivity(intent2);
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_review_details);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.adapter.ReviewDetailsAdapter.ReviewDelClickListener
    public void onReviewDelClick(int i) {
        topicReviewDel(Integer.valueOf(i));
    }
}
