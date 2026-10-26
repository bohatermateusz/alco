package com.hismith.hismithapp.fragment;

import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.CommunityAdapter;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.TopicListModel;
import com.hismith.hismithapp.model.TopicModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnLoadMoreListener;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PersonalMTopicFragment extends Fragment {
    public static final int WHAT_NOTIFYDATA = 1;
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private CommunityAdapter mAdapter;
    private LinearLayoutManager mLinearLayoutManager;
    private OnItemCountChangeListener mListener;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private List<TopicModel> mTopicModelList;
    private String userId;
    private int page = 1;
    private int totalPage = 1;
    public Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PersonalMTopicFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 1 && PersonalMTopicFragment.this.mRecyclerView != null) {
                PersonalMTopicFragment.this.mRecyclerView.scrollBy(0, 1);
                PersonalMTopicFragment.this.mRecyclerView.scrollBy(0, -1);
            }
            return false;
        }
    });

    public interface OnItemCountChangeListener {
        void getItemCount(String str);
    }

    public PersonalMTopicFragment() {
    }

    public PersonalMTopicFragment(String str) {
        this.userId = str;
    }

    static /* synthetic */ int access$208(PersonalMTopicFragment personalMTopicFragment) {
        int i = personalMTopicFragment.page;
        personalMTopicFragment.page = i + 1;
        return i;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getMyCollectTopic(int i) {
        BaseActivity baseActivity = this.baseActivity;
        if (baseActivity == null) {
            return;
        }
        this.baseRequest.getApi().myTopic((String) SharedPreferencesUtil.get(baseActivity, Constans.Token), "my", Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicListModel>(this.baseActivity, null, false) { // from class: com.hismith.hismithapp.fragment.PersonalMTopicFragment.4
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PersonalMTopicFragment.this.mRefreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PersonalMTopicFragment.this.mRefreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicListModel topicListModel) {
                if (topicListModel.getSuccess() == 1) {
                    if (topicListModel.getPage().intValue() == 1) {
                        PersonalMTopicFragment.this.mTopicModelList.clear();
                    }
                    PersonalMTopicFragment.this.totalPage = topicListModel.getTotalPage().intValue();
                    if (topicListModel.getTopics() != null) {
                        PersonalMTopicFragment.this.mTopicModelList.addAll(Arrays.asList(topicListModel.getTopics()));
                        if (PersonalMTopicFragment.this.mListener != null) {
                            PersonalMTopicFragment.this.mListener.getItemCount(topicListModel.getTotalCount() + "");
                        }
                        PersonalMTopicFragment.this.mAdapter.notifyDataSetChanged();
                    }
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getUserCollectTopic(int i) {
        BaseActivity baseActivity = this.baseActivity;
        if (baseActivity == null) {
            return;
        }
        this.baseRequest.getApi().userTopic((String) SharedPreferencesUtil.get(baseActivity, Constans.Token), this.userId, Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<TopicListModel>(this.baseActivity, null, false) { // from class: com.hismith.hismithapp.fragment.PersonalMTopicFragment.3
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PersonalMTopicFragment.this.mRefreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PersonalMTopicFragment.this.mRefreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(TopicListModel topicListModel) {
                if (topicListModel.getSuccess() == 1) {
                    if (topicListModel.getPage().intValue() == 1) {
                        PersonalMTopicFragment.this.mTopicModelList.clear();
                    }
                    PersonalMTopicFragment.this.totalPage = topicListModel.getTotalPage().intValue();
                    if (PersonalMTopicFragment.this.mListener != null) {
                        PersonalMTopicFragment.this.mListener.getItemCount(topicListModel.getTotalCount() + "");
                    }
                    if (topicListModel.getTopics() != null) {
                        PersonalMTopicFragment.this.mTopicModelList.addAll(Arrays.asList(topicListModel.getTopics()));
                        PersonalMTopicFragment.this.mAdapter.notifyDataSetChanged();
                    }
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        this.mRefreshLayout.setEnableRefresh(false);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.baseActivity);
        this.mLinearLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        this.mTopicModelList = new ArrayList();
        CommunityAdapter communityAdapter = new CommunityAdapter(this.baseActivity, this.mTopicModelList, false);
        this.mAdapter = communityAdapter;
        this.mRecyclerView.setAdapter(communityAdapter);
        this.page = 1;
        boolean isEmpty = TextUtils.isEmpty(this.userId);
        int i = this.page;
        if (isEmpty) {
            getMyCollectTopic(i);
        } else {
            getUserCollectTopic(i);
        }
    }

    private void initView(View view) {
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.personal_m_topic_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.personal_m_topic_recycler_view);
    }

    private void setListener() {
        this.mRefreshLayout.setOnLoadMoreListener(new OnLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.PersonalMTopicFragment.2
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int size = PersonalMTopicFragment.this.mTopicModelList.size();
                PersonalMTopicFragment personalMTopicFragment = PersonalMTopicFragment.this;
                if (size <= 0) {
                    personalMTopicFragment.page = 1;
                    boolean isEmpty = TextUtils.isEmpty(PersonalMTopicFragment.this.userId);
                    PersonalMTopicFragment personalMTopicFragment2 = PersonalMTopicFragment.this;
                    if (isEmpty) {
                        personalMTopicFragment2.getMyCollectTopic(personalMTopicFragment2.page);
                        return;
                    } else {
                        personalMTopicFragment2.getUserCollectTopic(personalMTopicFragment2.page);
                        return;
                    }
                }
                int i = personalMTopicFragment.page;
                int i2 = PersonalMTopicFragment.this.totalPage;
                PersonalMTopicFragment personalMTopicFragment3 = PersonalMTopicFragment.this;
                if (i >= i2) {
                    personalMTopicFragment3.mRefreshLayout.finishLoadMore(1000);
                    return;
                }
                PersonalMTopicFragment.access$208(personalMTopicFragment3);
                boolean isEmpty2 = TextUtils.isEmpty(PersonalMTopicFragment.this.userId);
                PersonalMTopicFragment personalMTopicFragment4 = PersonalMTopicFragment.this;
                if (isEmpty2) {
                    personalMTopicFragment4.getMyCollectTopic(personalMTopicFragment4.page);
                } else {
                    personalMTopicFragment4.getUserCollectTopic(personalMTopicFragment4.page);
                }
            }
        });
    }

    @Override // androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_m_topic, (ViewGroup) null);
        initView(inflate);
        init();
        setListener();
        return inflate;
    }

    public void onRefresh() {
        this.page = 1;
        boolean isEmpty = TextUtils.isEmpty(this.userId);
        int i = this.page;
        if (isEmpty) {
            getMyCollectTopic(i);
        } else {
            getUserCollectTopic(i);
        }
    }

    public void setmListener(OnItemCountChangeListener onItemCountChangeListener) {
        this.mListener = onItemCountChangeListener;
    }
}
