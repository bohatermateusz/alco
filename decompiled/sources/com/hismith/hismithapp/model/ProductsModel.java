package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;
import java.util.Arrays;

/* loaded from: classes3.dex */
public class ProductsModel extends HttpModel {
    private ProductModel[] products;

    public ProductModel[] getProducts() {
        return this.products;
    }

    public void setProducts(ProductModel[] productModelArr) {
        this.products = productModelArr;
    }

    public String toString() {
        return "ProductsModel{products=" + Arrays.toString(this.products) + '}';
    }
}
