package com.hismith.hismithapp.model;

import com.hismith.hismithapp.http.HttpModel;

/* loaded from: classes3.dex */
public class CountryModel extends HttpModel {
    private City[] countrys;

    public class City {
        private City[] citys;
        private Integer id;
        private String name;

        public City() {
        }

        public City(Integer num, String str) {
            this.id = num;
            this.name = str;
        }

        public City[] getCitys() {
            return this.citys;
        }

        public Integer getId() {
            return this.id;
        }

        public String getName() {
            return this.name;
        }

        public void setCitys(City[] cityArr) {
            this.citys = cityArr;
        }

        public void setId(Integer num) {
            this.id = num;
        }

        public void setName(String str) {
            this.name = str;
        }
    }

    public City[] getCountrys() {
        return this.countrys;
    }

    public void setCountrys(City[] cityArr) {
        this.countrys = cityArr;
    }
}
