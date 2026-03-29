from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    age: int | None = Field(default=None, description='Возраст человека')
    workclass: str | None = Field(default=None, description='Тип занятости')
    fnlwgt: int | None = Field(
        default=None,
        description='Вес наблюдения в данных переписи',
    )
    education: str | None = Field(default=None, description='Образование')
    education_num: int | None = Field(
        default=None,
        alias='education.num',
        description='Уровень образования в виде числа',
    )
    marital_status: str | None = Field(
        default=None,
        alias='marital.status',
        description='Семейное положение',
    )
    occupation: str | None = Field(
        default=None,
        description='Профессия / род деятельности',
    )
    relationship: str | None = Field(
        default=None,
        description='Роль человека в семье',
    )
    race: str | None = Field(default=None, description='Расовая группа')
    sex: str | None = Field(
        default=None,
        description='Пол человека (Male / Female)',
    )
    capital_gain: int | None = Field(
        default=None,
        alias='capital.gain',
        description='Доход от капитала (прибыль от продажи активов)',
    )
    capital_loss: int | None = Field(
        default=None,
        alias='capital.loss',
        description='Убытки от капитала',
    )
    hours_per_week: int | None = Field(
        default=None,
        alias='hours.per.week',
        description='Количество рабочих часов в неделю',
    )
    native_country: str | None = Field(
        default=None,
        alias='native.country',
        description='Страна происхождения',
    )

    class Config:
        populate_by_name = True


class PredictResponse(BaseModel):
    prediction: int = Field(description='Предсказанный класс')
    probability: float = Field(description='Вероятность предсказанного класса')


class UpdateModelRequest(BaseModel):
    run_id: str = Field(description='MLflow run_id')


class UpdateModelResponse(BaseModel):
    run_id: str = Field(description='MLflow run_id')

