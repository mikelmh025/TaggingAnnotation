import React, { useEffect, useState } from "react";
import "./mainMenu.styles.css";
import { useSelector } from 'react-redux';

const MainMenu = (props) => {
    const category = useSelector((state) => state.category.data);
    const results = useSelector((state) => state.results);
    const [list, setList] = useState(category);
    const [select, setSelect] = useState(props.selectCategory);
    const [selectIndex, setSelectIndex] = useState(props.groupIndex);
    const [selectItemIndex, setSelectItemIndex] = useState(props.optionIndex);
    const [mainImg, setMainImg] = useState('');

    const renderList = Object.keys(list).map((item, index) => {
        const renderItems = list[item].map((selection, itemIndex) => {
            return (
                <button
                    className='selectionButton'
                    key={itemIndex}
                    onClick={() => {
                        setSelect({ groupName: item, selection: selection.name });
                        props.onSelect({ groupName: item, selection: selection.name }, index, itemIndex);
                        setSelectIndex(index);
                        setSelectItemIndex(itemIndex);
                    }}
                    style={{
                        background: selection.check && "#78A55A",
                        color: selection.check && '#fff',
                        borderStyle: selectIndex === index && selectItemIndex === itemIndex && 'solid',
                    }}
                >
                    {selection.name}
                </button>
            );
        });
        return (
            <div className="groupContainer" key={index}>
                <h2 className="groupTitle">{item}: </h2>
                <div className="selectorContainer">
                    {renderItems}
                </div>
            </div>
        )
    });

    const checkAllCategory = () => {
        let allChecked = true;
        Object.keys(list).map((item) => {
            return list[item].map((element) => {
                if (!element.check) {
                    allChecked = false;
                }
                return allChecked;
            });
        })
        return allChecked;
    }

    useEffect(() => {
        setList(category);
    }, [category])

    useEffect(() => {
        setSelectIndex(props.groupIndex);
        setSelectItemIndex(props.optionIndex);
    }, [props.groupIndex, props.optionIndex]);

    window.addEventListener('message', function (e) {
        if (typeof e.data === 'string') {
            setMainImg(e.data);
        }
    })

    return (
        <div className="container">
            {mainImg ? 
                <img src={mainImg} className="image" alt="main to annotation" />
                :
                <div className="image"></div>
            }
            <div className="ListContainer">
                {renderList}
            </div>
            {checkAllCategory() ?
                <button className="submitButton"
                    onClick={() => {
                        const resultsJson = JSON.stringify(results.results)
                        window.parent.postMessage(resultsJson, '*')
                    }}>Submit</button> :
                <button className="submitButton" disabled>Submit</button>
            }

        </div>
    );
}

export default MainMenu;