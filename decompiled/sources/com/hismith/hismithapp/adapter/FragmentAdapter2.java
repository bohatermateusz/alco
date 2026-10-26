package com.hismith.hismithapp.adapter;

import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentActivity;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;

/* loaded from: classes3.dex */
public class FragmentAdapter2 extends FragmentStateAdapter {
    private List<Fragment> fragments;
    private List<String> titles;

    public FragmentAdapter2(FragmentActivity fragmentActivity) {
        super(fragmentActivity);
        this.fragments = new ArrayList();
        this.titles = new ArrayList();
    }

    static /* synthetic */ boolean lambda$containsItem$0(long j, Fragment fragment) {
        return ((long) fragment.hashCode()) == j;
    }

    @Override // androidx.viewpager2.adapter.FragmentStateAdapter
    public boolean containsItem(final long j) {
        List<Fragment> list = this.fragments;
        if (list == null) {
            return false;
        }
        return list.stream().filter(new Predicate() { // from class: com.hismith.hismithapp.adapter.FragmentAdapter2$$ExternalSyntheticLambda0
            @Override // java.util.function.Predicate
            public final boolean test(Object obj) {
                boolean nonNull;
                nonNull = Objects.nonNull((Fragment) obj);
                return nonNull;
            }
        }).anyMatch(new Predicate() { // from class: com.hismith.hismithapp.adapter.FragmentAdapter2$$ExternalSyntheticLambda1
            @Override // java.util.function.Predicate
            public final boolean test(Object obj) {
                return FragmentAdapter2.lambda$containsItem$0(j, (Fragment) obj);
            }
        });
    }

    @Override // androidx.viewpager2.adapter.FragmentStateAdapter
    public Fragment createFragment(int i) {
        return this.fragments.get(i);
    }

    @Override // androidx.recyclerview.widget.RecyclerView.Adapter
    public int getItemCount() {
        return this.fragments.size();
    }

    @Override // androidx.viewpager2.adapter.FragmentStateAdapter, androidx.recyclerview.widget.RecyclerView.Adapter
    public long getItemId(int i) {
        List<Fragment> list = this.fragments;
        if (list == null || i < 0 || i >= list.size()) {
            return super.getItemId(i);
        }
        return this.fragments.get(i) == null ? super.getItemId(i) : r0.hashCode();
    }

    public String getTitle(int i) {
        return this.titles.get(i);
    }

    public void updateData(List<Fragment> list, List<String> list2) {
        this.fragments = list;
        this.titles = list2;
        notifyDataSetChanged();
    }
}
