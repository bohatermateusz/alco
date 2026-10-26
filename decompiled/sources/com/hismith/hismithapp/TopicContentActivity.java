package com.hismith.hismithapp;

import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.Editable;
import android.text.Html;
import android.text.TextUtils;
import android.text.TextWatcher;
import android.view.KeyEvent;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;
import androidx.core.widget.NestedScrollView;
import androidx.fragment.app.FragmentActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.bumptech.glide.Glide;
import com.bumptech.glide.RequestBuilder;
import com.bumptech.glide.request.target.SimpleTarget;
import com.bumptech.glide.request.transition.Transition;
import com.google.firebase.messaging.ServiceStarter;
import com.hismith.hismithapp.adapter.TopicReviewAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.image_preview.ImagePreview;
import com.hismith.hismithapp.model.ReviewSendModel;
import com.hismith.hismithapp.model.TopicActionModel;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.model.TopicReviewModel;
import com.hismith.hismithapp.model.TopicReviewModel.Review;
import com.hismith.hismithapp.model.UserModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.LogUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.SoftKeyboardUtil;
import com.hismith.hismithapp.utils.TimeUtil;
import com.hismith.hismithapp.view.CircleImageView;
import com.hismith.hismithapp.view.RichText;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnLoadMoreListener;
import com.scwang.smart.refresh.layout.listener.OnRefreshListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/* loaded from: classes3.dex */
public class TopicContentActivity extends BaseActivity implements View.OnClickListener, OnLoadMoreListener {
    private RetrofitRequest baseRequest;
    private TopicReviewAdapter mAdapter;
    private ActivityResultLauncher<Intent> mBlockLauncher;
    private EditText mEditReview;
    private ImageView mImgCollect;
    private ImageView mImgDown;
    private CircleImageView mImgHead;
    private ImageView mImgLike;
    private ImageView mImgMore;
    private ImageView mImgReview;
    private ImageView mImgSearch;
    private ImageView mImgView;
    private LinearLayoutManager mLayoutManager;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private List<TopicReviewModel.Review> mReviewList;
    private NestedScrollView mScrollView;
    private ImageView mTitleBack;
    private TextView mTitleName;
    private TopicModel mTopicModel;
    private TextView mTxtCollect;
    private TextView mTxtCommentCount;
    private RichText mTxtContent;
    private TextView mTxtLike;
    private TextView mTxtName;
    private TextView mTxtPublish;
    private TextView mTxtReview;
    private TextView mTxtTime;
    private TextView mTxtTitle;
    private TextView mTxtView;
    private SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    private Integer topicId = 0;
    private int page = 1;
    private int totalPage = 1;
    private boolean isAction = false;
    private Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.TopicContentActivity.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 1) {
                TopicContentActivity.this.mScrollView.smoothScrollBy(0, TopicContentActivity.this.mRefreshLayout.getRefreshFooter().getView().getHeight());
            }
            return false;
        }
    });

    public static void MoveToPosition(LinearLayoutManager linearLayoutManager, RecyclerView recyclerView, int i) {
        int findFirstVisibleItemPosition = linearLayoutManager.findFirstVisibleItemPosition();
        int findLastVisibleItemPosition = linearLayoutManager.findLastVisibleItemPosition();
        if (i <= findFirstVisibleItemPosition) {
            recyclerView.scrollToPosition(i);
        } else if (i <= findLastVisibleItemPosition) {
            recyclerView.scrollBy(0, recyclerView.getChildAt(i - findFirstVisibleItemPosition).getTop());
        } else {
            recyclerView.scrollToPosition(i);
        }
    }

    private CharSequence formatUrlString(String str) {
        String[] linkString = linkString(str);
        String str2 = str.toString();
        if (linkString.length <= 0) {
            return null;
        }
        for (int i = 0; i < linkString.length; i++) {
            if (!TextUtils.isEmpty(linkString[i])) {
                str2 = str2.replace(linkString[i], "<font color='#48A7F9'> <a href=\"" + linkString[i] + "\">" + linkString[i] + "</a></font>");
            }
        }
        return Html.fromHtml(str2, 0);
    }

    private void getTopicDetails() {
        this.baseRequest.getApi().topicDetails((String) SharedPreferencesUtil.get(this, Constans.Token), this.topicId).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.TopicContentActivity.8
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicModel topicModel) {
                if (topicModel.getSuccess() == 1) {
                    TopicContentActivity.this.mTopicModel = topicModel;
                    Glide.with((FragmentActivity) TopicContentActivity.this).load(topicModel.getUserHeadIcon()).into((RequestBuilder<Drawable>) new SimpleTarget<Drawable>() { // from class: com.hismith.hismithapp.TopicContentActivity.8.1
                        public void onResourceReady(Drawable drawable, Transition<? super Drawable> transition) {
                            TopicContentActivity.this.mImgHead.setImageDrawable(drawable);
                        }

                        @Override // com.bumptech.glide.request.target.Target
                        public /* bridge */ /* synthetic */ void onResourceReady(Object obj, Transition transition) {
                            onResourceReady((Drawable) obj, (Transition<? super Drawable>) transition);
                        }
                    });
                    TopicContentActivity.this.mImgHead.setKing(topicModel.getUserKing(), TopicContentActivity.this, topicModel.getGradeIcon());
                    TopicContentActivity.this.mTxtTitle.setText(topicModel.getTitle());
                    TopicContentActivity.this.mTxtName.setText(topicModel.getUserName());
                    TopicContentActivity.this.mTxtTime.setText(TimeUtil.utc2Local(topicModel.getCreateDate()));
                    TopicContentActivity.this.mTxtLike.setText(topicModel.getLikeCount() + "");
                    TopicContentActivity.this.mTxtReview.setText(topicModel.getReviewCount() + "");
                    TopicContentActivity.this.mTxtCollect.setText(topicModel.getCollectCount() + "");
                    TopicContentActivity.this.mTxtView.setText(topicModel.getViewCount() + "");
                    int intValue = topicModel.getLikeed().intValue();
                    TopicContentActivity topicContentActivity = TopicContentActivity.this;
                    if (intValue == 1) {
                        topicContentActivity.mImgLike.setImageDrawable(ContextCompat.getDrawable(TopicContentActivity.this, R.mipmap.ic_community_like_pressed));
                        TopicContentActivity.this.mTxtLike.setTextColor(ContextCompat.getColor(TopicContentActivity.this, R.color.app_style_color));
                    } else {
                        topicContentActivity.mImgLike.setImageDrawable(ContextCompat.getDrawable(TopicContentActivity.this, R.mipmap.ic_community_like));
                        TopicContentActivity.this.mTxtLike.setTextColor(ContextCompat.getColor(TopicContentActivity.this, R.color.white80));
                    }
                    int intValue2 = topicModel.getCollected().intValue();
                    TopicContentActivity topicContentActivity2 = TopicContentActivity.this;
                    if (intValue2 == 1) {
                        topicContentActivity2.mImgCollect.setImageDrawable(ContextCompat.getDrawable(TopicContentActivity.this, R.mipmap.ic_community_collect_pressed));
                        TopicContentActivity.this.mTxtCollect.setTextColor(ContextCompat.getColor(TopicContentActivity.this, R.color.app_style_color));
                    } else {
                        topicContentActivity2.mImgCollect.setImageDrawable(ContextCompat.getDrawable(TopicContentActivity.this, R.mipmap.ic_community_collect));
                        TopicContentActivity.this.mTxtCollect.setTextColor(ContextCompat.getColor(TopicContentActivity.this, R.color.white80));
                    }
                    int intValue3 = topicModel.getViewed().intValue();
                    TopicContentActivity topicContentActivity3 = TopicContentActivity.this;
                    if (intValue3 == 1) {
                        topicContentActivity3.mImgView.setImageDrawable(ContextCompat.getDrawable(TopicContentActivity.this, R.mipmap.ic_community_read_pressed));
                        TopicContentActivity.this.mTxtView.setTextColor(ContextCompat.getColor(TopicContentActivity.this, R.color.app_style_color));
                    } else {
                        topicContentActivity3.mImgView.setImageDrawable(ContextCompat.getDrawable(TopicContentActivity.this, R.mipmap.ic_community_read));
                        TopicContentActivity.this.mTxtView.setTextColor(ContextCompat.getColor(TopicContentActivity.this, R.color.white80));
                    }
                    String content = topicModel.getContent();
                    TopicContentActivity.this.mTxtContent.setRichText(content.substring(content.indexOf("<div>"), content.lastIndexOf("</div>") + 6));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getTopicReviews(int i) {
        this.baseRequest.getApi().topicReviews((String) SharedPreferencesUtil.get(this, Constans.Token), this.topicId, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicReviewModel>(this, this.progressDialog, false) { // from class: com.hismith.hismithapp.TopicContentActivity.9
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                TopicContentActivity.this.mRefreshLayout.finishRefresh();
                TopicContentActivity.this.mRefreshLayout.finishLoadMore();
                if (TopicContentActivity.this.page >= TopicContentActivity.this.totalPage) {
                    TopicContentActivity.this.mRefreshLayout.setEnableLoadMore(false);
                }
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                TopicContentActivity.this.mRefreshLayout.finishRefresh();
                TopicContentActivity.this.mRefreshLayout.finishLoadMore();
                if (TopicContentActivity.this.page >= TopicContentActivity.this.totalPage) {
                    TopicContentActivity.this.mRefreshLayout.setEnableLoadMore(false);
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicReviewModel topicReviewModel) {
                if (topicReviewModel.getSuccess() == 1) {
                    TopicContentActivity.this.page = topicReviewModel.getPage().intValue();
                    TopicContentActivity.this.totalPage = topicReviewModel.getTotalPage().intValue();
                    if (TopicContentActivity.this.page <= 1) {
                        TopicContentActivity.this.mReviewList.clear();
                    }
                    if (topicReviewModel.getReviews() != null) {
                        TopicContentActivity.this.mReviewList.addAll(Arrays.asList(topicReviewModel.getReviews()));
                        TopicContentActivity.this.mAdapter.notifyDataSetChanged();
                        TopicContentActivity.this.mHandler.sendEmptyMessageDelayed(1, 200L);
                    }
                    if (TopicContentActivity.this.page < TopicContentActivity.this.totalPage) {
                        TopicContentActivity.this.mRefreshLayout.setEnableLoadMore(true);
                        SmartRefreshLayout smartRefreshLayout = TopicContentActivity.this.mRefreshLayout;
                        final TopicContentActivity topicContentActivity = TopicContentActivity.this;
                        smartRefreshLayout.setOnLoadMoreListener(new OnLoadMoreListener() { // from class: com.hismith.hismithapp.TopicContentActivity$9$$ExternalSyntheticLambda0
                            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
                            public final void onLoadMore(RefreshLayout refreshLayout) {
                                TopicContentActivity.this.onLoadMore(refreshLayout);
                            }
                        });
                    }
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.topicId = Integer.valueOf(getIntent().getIntExtra("topicId", 0));
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this);
        this.mLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mReviewList = new ArrayList();
        TopicReviewAdapter topicReviewAdapter = new TopicReviewAdapter(this, this.topicId, this.mReviewList);
        this.mAdapter = topicReviewAdapter;
        this.mRecyclerView.setAdapter(topicReviewAdapter);
        this.mRefreshLayout.setEnableRefresh(false);
        this.mRefreshLayout.setEnableLoadMore(false);
        initActivityResult();
        getTopicDetails();
        getTopicReviews(this.page);
    }

    private void initActivityResult() {
        this.mBlockLauncher = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() { // from class: com.hismith.hismithapp.TopicContentActivity.7
            @Override // androidx.activity.result.ActivityResultCallback
            public void onActivityResult(ActivityResult activityResult) {
                Intent data;
                LogUtil.e("onActivityResult:" + activityResult.getResultCode());
                if (activityResult.getResultCode() != 108 || (data = activityResult.getData()) == null) {
                    return;
                }
                int intExtra = data.getIntExtra(Constans.BlockUserId, 0);
                LogUtil.e("onActivityResult 返回的拉黑UserId:" + intExtra);
                int intValue = TopicContentActivity.this.mTopicModel.getUserId().intValue();
                TopicContentActivity topicContentActivity = TopicContentActivity.this;
                if (intExtra == intValue) {
                    topicContentActivity.setResult(108);
                    TopicContentActivity.this.appManager.finishActivity();
                    return;
                }
                topicContentActivity.mReviewList.clear();
                if (TopicContentActivity.this.mAdapter != null) {
                    TopicContentActivity.this.mAdapter.notifyDataSetChanged();
                }
                TopicContentActivity.this.page = 1;
                TopicContentActivity.this.totalPage = 1;
                TopicContentActivity topicContentActivity2 = TopicContentActivity.this;
                topicContentActivity2.getTopicReviews(topicContentActivity2.page);
            }
        });
    }

    private void initView() {
        this.mTitleBack = (ImageView) findViewById(R.id.title_img_back);
        this.mTitleName = (TextView) findViewById(R.id.title_txt_name);
        this.mScrollView = (NestedScrollView) findViewById(R.id.content_scrollView);
        this.mImgSearch = (ImageView) findViewById(R.id.content_img_search);
        this.mImgMore = (ImageView) findViewById(R.id.content_img_more);
        this.mTxtTitle = (TextView) findViewById(R.id.content_txt_title);
        this.mTxtName = (TextView) findViewById(R.id.content_txt_name);
        this.mTxtTime = (TextView) findViewById(R.id.content_txt_time);
        this.mImgHead = (CircleImageView) findViewById(R.id.content_img_head);
        this.mImgDown = (ImageView) findViewById(R.id.content_img_down);
        this.mTxtContent = (RichText) findViewById(R.id.content_txt_content);
        this.mImgLike = (ImageView) findViewById(R.id.content_img_like);
        this.mImgReview = (ImageView) findViewById(R.id.content_img_review);
        this.mImgCollect = (ImageView) findViewById(R.id.content_img_collect);
        this.mImgView = (ImageView) findViewById(R.id.content_img_view);
        this.mTxtLike = (TextView) findViewById(R.id.content_txt_like);
        this.mTxtReview = (TextView) findViewById(R.id.content_txt_review);
        this.mTxtCollect = (TextView) findViewById(R.id.content_txt_collect);
        this.mTxtView = (TextView) findViewById(R.id.content_txt_view);
        this.mRefreshLayout = (SmartRefreshLayout) findViewById(R.id.content_refreshLayout);
        this.mRecyclerView = (RecyclerView) findViewById(R.id.content_recycler);
        this.mEditReview = (EditText) findViewById(R.id.content_edit_review);
        this.mTxtPublish = (TextView) findViewById(R.id.content_txt_publish);
        this.mTxtCommentCount = (TextView) findViewById(R.id.content_txt_comment_count);
        LanguageUtil.setText(this, this.mTxtPublish, "publish");
    }

    private void setListener() {
        this.mTitleBack.setOnClickListener(this);
        this.mImgSearch.setOnClickListener(this);
        this.mImgHead.setOnClickListener(this);
        this.mTxtName.setOnClickListener(this);
        this.mImgMore.setOnClickListener(this);
        this.mImgDown.setOnClickListener(this);
        this.mImgLike.setOnClickListener(this);
        this.mTxtLike.setOnClickListener(this);
        this.mImgReview.setOnClickListener(this);
        this.mTxtReview.setOnClickListener(this);
        this.mImgCollect.setOnClickListener(this);
        this.mTxtCollect.setOnClickListener(this);
        this.mTxtPublish.setOnClickListener(this);
        this.mTxtPublish.setClickable(false);
        this.mAdapter.setOnUserInfoListener(new TopicReviewAdapter.OnUserInfoListener() { // from class: com.hismith.hismithapp.TopicContentActivity.2
            @Override // com.hismith.hismithapp.adapter.TopicReviewAdapter.OnUserInfoListener
            public void onUserInfoClick(int i) {
                if (MyApplication.getApplication().getmUserModel() != null) {
                    if (i == MyApplication.getApplication().getmUserModel().getUserId()) {
                        TopicContentActivity.this.startActivity(new Intent(TopicContentActivity.this, (Class<?>) PersonalDetailsActivity2.class));
                    } else {
                        Intent intent = new Intent(TopicContentActivity.this, (Class<?>) UserDetailsActivity2.class);
                        intent.putExtra("userId", i);
                        TopicContentActivity.this.mBlockLauncher.launch(intent);
                    }
                }
            }
        });
        this.mRefreshLayout.setOnRefreshListener(new OnRefreshListener() { // from class: com.hismith.hismithapp.TopicContentActivity.3
            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                LogUtil.e("刷新列表onRefresh");
                TopicContentActivity.this.page = 1;
                TopicContentActivity.this.totalPage = 1;
                TopicContentActivity topicContentActivity = TopicContentActivity.this;
                topicContentActivity.getTopicReviews(topicContentActivity.page);
            }
        });
        this.mEditReview.addTextChangedListener(new TextWatcher() { // from class: com.hismith.hismithapp.TopicContentActivity.4
            @Override // android.text.TextWatcher
            public void afterTextChanged(Editable editable) {
            }

            @Override // android.text.TextWatcher
            public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3) {
            }

            @Override // android.text.TextWatcher
            public void onTextChanged(CharSequence charSequence, int i, int i2, int i3) {
                boolean isEmpty = TextUtils.isEmpty(charSequence);
                TopicContentActivity topicContentActivity = TopicContentActivity.this;
                if (isEmpty) {
                    topicContentActivity.mTxtPublish.setTextColor(TopicContentActivity.this.getResources().getColor(R.color.gray));
                    TopicContentActivity.this.mTxtPublish.setClickable(false);
                } else {
                    topicContentActivity.mTxtPublish.setTextColor(TopicContentActivity.this.getResources().getColor(R.color.app_style_color));
                    TopicContentActivity.this.mTxtPublish.setClickable(true);
                }
            }
        });
        this.mEditReview.setOnEditorActionListener(new TextView.OnEditorActionListener() { // from class: com.hismith.hismithapp.TopicContentActivity.5
            @Override // android.widget.TextView.OnEditorActionListener
            public boolean onEditorAction(TextView textView, int i, KeyEvent keyEvent) {
                if (i != 4) {
                    return false;
                }
                TopicContentActivity.this.topicReviewSend();
                SoftKeyboardUtil.hideSoftKeyboard(TopicContentActivity.this, textView);
                return false;
            }
        });
        this.mTxtContent.setOnRichTextImageClickListener(new RichText.OnRichTextImageClickListener() { // from class: com.hismith.hismithapp.TopicContentActivity.6
            @Override // com.hismith.hismithapp.view.RichText.OnRichTextImageClickListener
            public void imageClicked(List<String> list, int i) {
                ImagePreview.getInstance().setContext(TopicContentActivity.this).setTopicId(TopicContentActivity.this.topicId.intValue()).setIndex(i).setImageList(list).setFolderName("hismith/image").setEnableDragClose(true).start();
            }
        });
    }

    private void topicCollectAction(final String str) {
        this.isAction = true;
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this, Constans.Token), this.topicId, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicActionModel>(this, null, false) { // from class: com.hismith.hismithapp.TopicContentActivity.11
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                TopicContentActivity.this.isAction = false;
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                TopicContentActivity.this.isAction = false;
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicActionModel topicActionModel) {
                if (topicActionModel.getSuccess() == 1) {
                    String collectCount = topicActionModel.getCollectCount();
                    if ("collect".equals(str)) {
                        TopicContentActivity.this.mTopicModel.setCollected(1);
                    } else if ("uncollect".equals(str)) {
                        TopicContentActivity.this.mTopicModel.setCollected(0);
                    }
                    TopicContentActivity.this.mTopicModel.setCollectCount(collectCount);
                    TopicContentActivity.this.mTxtCollect.setText(TopicContentActivity.this.mTopicModel.getCollectCount() + "");
                    int intValue = TopicContentActivity.this.mTopicModel.getCollected().intValue();
                    TopicContentActivity topicContentActivity = TopicContentActivity.this;
                    if (intValue == 1) {
                        topicContentActivity.mImgCollect.setImageDrawable(ContextCompat.getDrawable(TopicContentActivity.this, R.mipmap.ic_community_collect_pressed));
                        TopicContentActivity.this.mTxtCollect.setTextColor(ContextCompat.getColor(TopicContentActivity.this, R.color.app_style_color));
                    } else {
                        topicContentActivity.mImgCollect.setImageDrawable(ContextCompat.getDrawable(TopicContentActivity.this, R.mipmap.ic_community_collect));
                        TopicContentActivity.this.mTxtCollect.setTextColor(ContextCompat.getColor(TopicContentActivity.this, R.color.white80));
                    }
                }
            }
        });
    }

    private void topicLikeAction(final String str) {
        this.isAction = true;
        this.baseRequest.getApi().topicAction((String) SharedPreferencesUtil.get(this, Constans.Token), this.topicId, str).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicActionModel>(this, null, false) { // from class: com.hismith.hismithapp.TopicContentActivity.10
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                TopicContentActivity.this.isAction = false;
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                TopicContentActivity.this.isAction = false;
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicActionModel topicActionModel) {
                if (topicActionModel.getSuccess() == 1) {
                    String likeCount = topicActionModel.getLikeCount();
                    if ("like".equals(str)) {
                        TopicContentActivity.this.mTopicModel.setLikeed(1);
                    } else if ("unlike".equals(str)) {
                        TopicContentActivity.this.mTopicModel.setLikeed(0);
                    }
                    TopicContentActivity.this.mTopicModel.setLikeCount(likeCount);
                    TopicContentActivity.this.mTxtLike.setText(TopicContentActivity.this.mTopicModel.getLikeCount() + "");
                    int intValue = TopicContentActivity.this.mTopicModel.getLikeed().intValue();
                    TopicContentActivity topicContentActivity = TopicContentActivity.this;
                    if (intValue == 1) {
                        topicContentActivity.mImgLike.setImageDrawable(ContextCompat.getDrawable(TopicContentActivity.this, R.mipmap.ic_community_like_pressed));
                        TopicContentActivity.this.mTxtLike.setTextColor(ContextCompat.getColor(TopicContentActivity.this, R.color.app_style_color));
                    } else {
                        topicContentActivity.mImgLike.setImageDrawable(ContextCompat.getDrawable(TopicContentActivity.this, R.mipmap.ic_community_like));
                        TopicContentActivity.this.mTxtLike.setTextColor(ContextCompat.getColor(TopicContentActivity.this, R.color.white80));
                    }
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void topicReviewSend() {
        String str = (String) SharedPreferencesUtil.get(this, Constans.Token);
        final String obj = this.mEditReview.getText().toString();
        if (TextUtils.isEmpty(obj)) {
            return;
        }
        this.baseRequest.getApi().topicReviewSend(str, this.topicId, obj).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<ReviewSendModel>(this, this.progressDialog) { // from class: com.hismith.hismithapp.TopicContentActivity.12
            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(ReviewSendModel reviewSendModel) {
                if (reviewSendModel.getSuccess() == 1) {
                    TopicContentActivity.this.mEditReview.setText("");
                    TopicReviewModel topicReviewModel = new TopicReviewModel();
                    Objects.requireNonNull(topicReviewModel);
                    TopicReviewModel.Review review = topicReviewModel.new Review();
                    review.setId(reviewSendModel.getRid());
                    review.setLikeed(0);
                    review.setLikeCount(0);
                    review.setContent(obj);
                    review.setCreateDate(TopicContentActivity.this.sdf.format(new Date()));
                    review.setReplyCount(0);
                    UserModel userModel = MyApplication.getApplication().getmUserModel();
                    if (userModel != null) {
                        review.setUserGender(Integer.valueOf(userModel.getGender()));
                        review.setUserHeadIcon(userModel.getHeadIcon());
                        review.setUserId(Integer.valueOf(userModel.getUserId()));
                        review.setUserKing(Integer.valueOf(userModel.getKing()));
                        review.setUserName(userModel.getName());
                    }
                    TopicContentActivity.this.mReviewList.add(0, review);
                    TopicContentActivity.this.mAdapter.notifyDataSetChanged();
                }
            }
        });
    }

    public String[] linkString(String str) {
        Matcher matcher = Pattern.compile("(((https|http)?://)?([a-z0-9]+[.])|(www.))\\w+[.|\\/]([a-z0-9]{0,})?[[.]([a-z0-9]{0,})]+((/[\\S&&[^,;一-龥]]+)+)?([.][a-z0-9]{0,}+|/?)").matcher(str);
        String[] strArr = new String[str.length() / 5];
        int i = 0;
        while (matcher.find()) {
            strArr[i] = matcher.group();
            i++;
        }
        return strArr;
    }

    @Override // android.view.View.OnClickListener
    public void onClick(View view) {
        TopicModel topicModel;
        int id = view.getId();
        if (id == R.id.title_img_back) {
            finish();
            return;
        }
        if (id == R.id.content_img_search) {
            return;
        }
        if (id == R.id.content_img_head || id == R.id.content_txt_name) {
            if (MyApplication.getApplication().getmUserModel() == null || (topicModel = this.mTopicModel) == null) {
                return;
            }
            if (topicModel.getUserId().intValue() == MyApplication.getApplication().getmUserModel().getUserId()) {
                startActivity(new Intent(this, (Class<?>) PersonalDetailsActivity2.class));
                return;
            }
            Intent intent = new Intent(this, (Class<?>) UserDetailsActivity2.class);
            intent.putExtra("userId", this.mTopicModel.getUserId());
            this.mBlockLauncher.launch(intent);
            return;
        }
        if (id == R.id.content_img_more || id == R.id.content_img_down) {
            return;
        }
        if (id == R.id.content_img_like || id == R.id.content_txt_like) {
            TopicModel topicModel2 = this.mTopicModel;
            if (topicModel2 == null || this.isAction) {
                return;
            }
            topicLikeAction(topicModel2.getLikeed().intValue() == 1 ? "unlike" : "like");
            return;
        }
        if (id == R.id.content_img_review || id == R.id.content_txt_review) {
            return;
        }
        if (id != R.id.content_img_collect && id != R.id.content_txt_collect) {
            if (id == R.id.content_txt_publish) {
                topicReviewSend();
            }
        } else {
            TopicModel topicModel3 = this.mTopicModel;
            if (topicModel3 == null || this.isAction) {
                return;
            }
            topicCollectAction(topicModel3.getCollected().intValue() == 1 ? "uncollect" : "collect");
        }
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.fragment.app.FragmentActivity, androidx.activity.ComponentActivity, androidx.core.app.ComponentActivity, android.app.Activity
    protected void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        setContentView(R.layout.activity_topic_content);
        initView();
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.BaseActivity, androidx.appcompat.app.AppCompatActivity, androidx.fragment.app.FragmentActivity, android.app.Activity
    protected void onDestroy() {
        super.onDestroy();
    }

    @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
    public void onLoadMore(RefreshLayout refreshLayout) {
        LogUtil.e("加载列表onLoadMore");
        int i = this.page;
        if (i >= this.totalPage) {
            this.mRefreshLayout.finishLoadMore(ServiceStarter.ERROR_UNKNOWN);
            return;
        }
        int i2 = i + 1;
        this.page = i2;
        getTopicReviews(i2);
    }
}
