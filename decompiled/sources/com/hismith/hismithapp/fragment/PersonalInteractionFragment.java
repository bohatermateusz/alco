package com.hismith.hismithapp.fragment;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.hismith.hismithapp.BaseActivity;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.adapter.InteractionAdapter;
import com.hismith.hismithapp.application.MyApplication;
import com.hismith.hismithapp.http.RetrofitRequest;
import com.hismith.hismithapp.http.RxSubscribe;
import com.hismith.hismithapp.model.FollowModel;
import com.hismith.hismithapp.model.InteractiveModel;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;
import com.hismith.hismithapp.utils.ToastUtil;
import com.scwang.smart.refresh.layout.SmartRefreshLayout;
import com.scwang.smart.refresh.layout.api.RefreshLayout;
import com.scwang.smart.refresh.layout.listener.OnRefreshLoadMoreListener;
import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.schedulers.Schedulers;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes3.dex */
public class PersonalInteractionFragment extends LazyLoadFragment {
    public static final int WHAT_NOTIFYDATA = 1;
    private BaseActivity baseActivity;
    private RetrofitRequest baseRequest;
    private InteractionAdapter mAdapter;
    private LinearLayoutManager mLinearLayoutManager;
    private RecyclerView mRecyclerView;
    private SmartRefreshLayout mRefreshLayout;
    private List<InteractiveModel.Interactive> mInteractiveList = new ArrayList();
    private int lastInteractionId = -1;
    private int page = 1;
    private int totalPage = 1;
    public Handler mHandler = new Handler(new Handler.Callback() { // from class: com.hismith.hismithapp.fragment.PersonalInteractionFragment.1
        @Override // android.os.Handler.Callback
        public boolean handleMessage(Message message) {
            if (message.what == 1 && PersonalInteractionFragment.this.mRecyclerView != null) {
                PersonalInteractionFragment.this.mRecyclerView.scrollBy(0, 1);
                PersonalInteractionFragment.this.mRecyclerView.scrollBy(0, -1);
            }
            return false;
        }
    });

    static /* synthetic */ int access$108(PersonalInteractionFragment personalInteractionFragment) {
        int i = personalInteractionFragment.page;
        personalInteractionFragment.page = i + 1;
        return i;
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void follow(final Integer num) {
        Observable<FollowModel> observeOn = this.baseRequest.getApi().follow((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), Integer.valueOf(this.mInteractiveList.get(num.intValue()).getUserId()), Integer.valueOf(this.mInteractiveList.get(num.intValue()).getFollowed().intValue() == 0 ? 1 : 0)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread());
        BaseActivity baseActivity = this.baseActivity;
        observeOn.subscribe(new RxSubscribe<FollowModel>(baseActivity, baseActivity.progressDialog, false) { // from class: com.hismith.hismithapp.fragment.PersonalInteractionFragment.5
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PersonalInteractionFragment.this.mAdapter.notifyItemChanged(num.intValue(), "");
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PersonalInteractionFragment.this.mAdapter.notifyItemChanged(num.intValue(), "");
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(FollowModel followModel) {
                if (followModel.getSuccess() == 1) {
                    ((InteractiveModel.Interactive) PersonalInteractionFragment.this.mInteractiveList.get(num.intValue())).setFollowed(followModel.getFollow());
                    ((InteractiveModel.Interactive) PersonalInteractionFragment.this.mInteractiveList.get(num.intValue())).setStatus(0);
                    if (followModel.getFollow().intValue() == 1) {
                        if (MyApplication.getApplication().getmUserModel() != null) {
                            MyApplication.getApplication().getmUserModel().setFocusCount(MyApplication.getApplication().getmUserModel().getFocusCount() + 1);
                            PersonalInteractionFragment.this.baseActivity.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                        }
                        ToastUtil.showShortToastCenter(PersonalInteractionFragment.this.baseActivity, LanguageUtil.getLanguageByKey(PersonalInteractionFragment.this.baseActivity, "followedMsg"));
                        return;
                    }
                    if (MyApplication.getApplication().getmUserModel() != null) {
                        MyApplication.getApplication().getmUserModel().setFocusCount(MyApplication.getApplication().getmUserModel().getFocusCount() - 1);
                        PersonalInteractionFragment.this.baseActivity.sendBroadcast(new Intent(Constans.RECEIVER_USER_INFO_CHANGE));
                    }
                    ToastUtil.showShortToastCenter(PersonalInteractionFragment.this.baseActivity, LanguageUtil.getLanguageByKey(PersonalInteractionFragment.this.baseActivity, "unfollowedMsg"));
                }
            }
        });
    }

    /* JADX INFO: Access modifiers changed from: private */
    public void getMsgInteractive(int i) {
        this.baseRequest.getApi().msgInteractive((String) SharedPreferencesUtil.get(this.baseActivity, Constans.Token), Integer.valueOf(i)).subscribeOn(Schedulers.io()).observeOn(AndroidSchedulers.mainThread()).subscribe(new RxSubscribe<InteractiveModel>(this.baseActivity, null, false) { // from class: com.hismith.hismithapp.fragment.PersonalInteractionFragment.4
            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onComplete() {
                super.onComplete();
                PersonalInteractionFragment.this.mRefreshLayout.finishRefresh();
                PersonalInteractionFragment.this.mRefreshLayout.finishLoadMore();
            }

            @Override // com.hismith.hismithapp.http.RxSubscribe, io.reactivex.Observer
            public void onError(Throwable th) {
                super.onError(th);
                PersonalInteractionFragment.this.mRefreshLayout.finishRefresh();
                PersonalInteractionFragment.this.mRefreshLayout.finishLoadMore();
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.hismith.hismithapp.http.RxSubscribe
            public void onSuccess(InteractiveModel interactiveModel) {
                if (interactiveModel.getSuccess() == 1) {
                    if (interactiveModel.getPage() == 1) {
                        PersonalInteractionFragment.this.mInteractiveList.clear();
                    }
                    PersonalInteractionFragment.this.totalPage = interactiveModel.getTotalPage();
                    if (interactiveModel.getInteractive() != null) {
                        PersonalInteractionFragment.this.mInteractiveList.addAll(Arrays.asList(interactiveModel.getInteractive()));
                        PersonalInteractionFragment.this.mAdapter.notifyDataSetChanged();
                    }
                }
            }
        });
    }

    private void init() {
        this.baseRequest = new RetrofitRequest();
        int intValue = ((Integer) SharedPreferencesUtil.get(this.baseActivity, Constans.LastInteractionId, -1)).intValue();
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(this.baseActivity);
        this.mLinearLayoutManager = linearLayoutManager;
        this.mRecyclerView.setLayoutManager(linearLayoutManager);
        InteractionAdapter interactionAdapter = new InteractionAdapter(this.baseActivity, this.mInteractiveList, intValue);
        this.mAdapter = interactionAdapter;
        this.mRecyclerView.setAdapter(interactionAdapter);
        this.mRefreshLayout.autoRefresh();
    }

    private void initView(View view) {
        this.mRefreshLayout = (SmartRefreshLayout) view.findViewById(R.id.personal_interaction_refreshLayout);
        this.mRecyclerView = (RecyclerView) view.findViewById(R.id.personal_interaction_recycler_view);
    }

    private void setListener() {
        this.mRefreshLayout.setOnRefreshLoadMoreListener(new OnRefreshLoadMoreListener() { // from class: com.hismith.hismithapp.fragment.PersonalInteractionFragment.2
            @Override // com.scwang.smart.refresh.layout.listener.OnLoadMoreListener
            public void onLoadMore(RefreshLayout refreshLayout) {
                int i = PersonalInteractionFragment.this.page;
                int i2 = PersonalInteractionFragment.this.totalPage;
                PersonalInteractionFragment personalInteractionFragment = PersonalInteractionFragment.this;
                if (i >= i2) {
                    personalInteractionFragment.mRefreshLayout.finishLoadMore(1000);
                    return;
                }
                PersonalInteractionFragment.access$108(personalInteractionFragment);
                PersonalInteractionFragment personalInteractionFragment2 = PersonalInteractionFragment.this;
                personalInteractionFragment2.getMsgInteractive(personalInteractionFragment2.page);
            }

            @Override // com.scwang.smart.refresh.layout.listener.OnRefreshListener
            public void onRefresh(RefreshLayout refreshLayout) {
                PersonalInteractionFragment.this.page = 1;
                PersonalInteractionFragment personalInteractionFragment = PersonalInteractionFragment.this;
                personalInteractionFragment.getMsgInteractive(personalInteractionFragment.page);
            }
        });
        this.mAdapter.setOnItemFollowClickListener(new InteractionAdapter.OnItemFollowClickListener() { // from class: com.hismith.hismithapp.fragment.PersonalInteractionFragment.3
            @Override // com.hismith.hismithapp.adapter.InteractionAdapter.OnItemFollowClickListener
            public void onItemFollow(int i) {
                PersonalInteractionFragment.this.follow(Integer.valueOf(i));
            }
        });
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    protected void lazyLoad() {
        init();
        setListener();
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment
    public void loadData() {
        SmartRefreshLayout smartRefreshLayout = this.mRefreshLayout;
        if (smartRefreshLayout != null) {
            smartRefreshLayout.autoRefresh();
        }
    }

    @Override // com.hismith.hismithapp.fragment.LazyLoadFragment, androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.baseActivity = (BaseActivity) getActivity();
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        View inflate = layoutInflater.inflate(R.layout.fragment_personal_interaction, (ViewGroup) null);
        initView(inflate);
        return inflate;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        List<InteractiveModel.Interactive> list = this.mInteractiveList;
        if (list == null || list.size() <= 1) {
            return;
        }
        SharedPreferencesUtil.put(this.baseActivity, Constans.LastInteractionId, Integer.valueOf(this.mInteractiveList.get(0).getId()));
    }
}
