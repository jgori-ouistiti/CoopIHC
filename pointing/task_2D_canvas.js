
            // ========================== functions and classes def =============

            function getRandomFromBucket(bucket) {
               var randomIndex = Math.floor(Math.random()*bucket.length);
               return bucket.splice(randomIndex, 1)[0];
            }

            function index_to_coords(index, gridsize){
                var nl = gridsize[0],
                    nc =  gridsize[1];
                var ci = index%nc;
                var li = (index-ci)/nc;
                return [li,ci];

            }
            function coords_to_index(coords, gridsize){
                var li = coords[0],
                    ci = coords[1],
                    nl = gridsize[0],
                    nc = gridsize[1];
                return li*nc + ci;


            }

            function getRandomTargets(pointing_task_parametersObject){
                var bucket = [];
                var gridindex = pointing_task_parametersObject.gridsize[0]*pointing_task_parametersObject.gridsize[1];
                for (var i=0;i<gridindex;i++) {
                    bucket.push(i);
                }
                var targetArray_index = new Array(pointing_task_parametersObject.number_of_targets + 1);
                var targetArray = new Array(pointing_task_parametersObject.number_of_targets + 1);

                for (var i=0;i<pointing_task_parametersObject.number_of_targets+1;i++){
                    var random_index = getRandomFromBucket(bucket)
                    targetArray_index[i] = random_index;
                    targetArray[i] = index_to_coords(random_index, pointing_task_parametersObject.gridsize );
                }
                let position = index_to_coords(targetArray_index.pop(), pointing_task_parametersObject.gridsize );
                // Don't forget to pop the targetArray as well
                targetArray.pop();
                let randomIndex = Math.floor(Math.random()*targetArray_index.length);
                let goal = index_to_coords(targetArray_index[randomIndex], pointing_task_parametersObject.gridsize);

                return [position, targetArray.sort(function(a,b){return coords_to_index(a,pointing_task_parametersObject.gridsize )-coords_to_index(b,pointing_task_parametersObject.gridsize);}), goal];
            }

            class PointingTaskParameters{
                constructor(gridsize = [31,31], number_of_targets = 8, mode = 'gain'){
                    this.gridsize = gridsize;
                    this.number_of_targets = number_of_targets;
                    this.mode = mode;
                }

                form_string(){
                    var _str = '-'
                    for (const item in this){
                        _str += item.toString() + ':' + this[item].toString() + '-';
                    }
                    return _str;
                }
            }

            class PointingTaskManager{
                constructor(){
                    this.__state = null;
                    this.__position_text = document.querySelector('.position');
                    this.__targets_text = document.querySelector('.targets');
                    this.__position = null;
                    this.__targets = null;
                    this.__user_action_text = document.querySelector('.user_action');
                    this.__user_action = null;
                    this.__assistant_action_text = document.querySelector('.assistant_action');
                    this.__assistant_action = null;

                }

                get state(){
                    return {'type': 'task_state', 'position': this.position, 'targets': this.targets};
                }

                get position(){
                    return this.__position;
                }

                get targets(){
                    return this.__targets;
                }

                set position(value){
                    this.__position = value;
                    this.__position_text.textContent = "["+value.toString()+"]";
                }

                set targets(value){
                    this.__targets = value;
                    this.__targets_text.textContent = "["+ value.join('] [') + ']';
                    // this.__targets_text.textContent = value;

                }

                get user_action(){
                    return this.__user_action;
                }

                set user_action(value){
                    this.__user_action = value;
                    this.__user_action_text.textContent = value.toString();
                }

                get assistant_action(){
                    return this.__assistant_action;
                }

                set assistant_action(value){
                    this.__assistant_action = value;
                    this.__assistant_action_text.textContent = value.toString();
                }



            }


            function send_msg(msg) {
                websocket.send(JSON.stringify(msg))
            }

            function update_state(received_dic){
                for (const [key, value] of Object.entries(received_dic)){
                        document.querySelector('.' + key).textContent = value;
                }

            }

            var pointing_task_parameters = new PointingTaskParameters();
            var pointing_task_manager = new PointingTaskManager();




            function draw(){
                var canvas = document.getElementById('render-canvas');
                var context = canvas.getContext("2d");
            }







            function on_init(received_dic){
                let params = received_dic['parameters'];
                for (const [key, value] of Object.entries(params)){
                    pointing_task_parameters[key] = value;
                }
                parameters.textContent = pointing_task_parameters.form_string();
                self_reset();
            }

            function self_reset(){
                let [_position, _targets, _goal] = getRandomTargets(pointing_task_parameters);
                pointing_task_manager.position = _position;
                pointing_task_manager.targets = _targets;
                console.log(pointing_task_manager);
                send_msg(pointing_task_manager.state);
            }

            function on_reset(received_dic){
                if (received_dic['reset_dic'] == null){
                    self_reset()

                }
                else {
                    position = received_dic['reset_dic']['position'];
                    targets = received_dic['reset_dic']['targets'];
                    // Why does this not work?
                    if (!(position == pointing_task_manager.position && targets == pointing_task_manager.targets)){
                        console.log('warning, forced reset with ', position, 'versus', pointing_task_manager.position, 'and', targets, 'versus', pointing_task_manager.targets);
                        pointing_task_manager.position = position;
                        pointing_task_manager.targets = targets;
                    send_msg(pointing_task_manager.state);
                    }
                }

            }

            function unpack_actions(received_action_dic){
                action = received_action_dic["value"];
                action_value = action['values'];
                action_possible_values = action['possible_values'];
                action_array = Array();
                for (let i=0; i<action_value.length; i++){
                    // can't make the test for null || [null] work
                    if (action_possible_values[i][action_value[i]] == undefined){
                        action_array.push(action_value[i]);
                    }
                    else{
                        action_array.push(action_possible_values[i][action_value[i]]);
                    }

                }
                return action_array;
            }

            function on_user_action(received_dic){

                var is_done = false;

                // convert action
                action = unpack_actions(received_dic);
                console.log("user_action", action);
                pointing_task_manager.user_action = action;
                if (action == 0){
                    is_done = true;
                }
                return_dic = {"state": pointing_task_manager.state, "reward": -.5, "is_done": is_done};
                send_msg(return_dic);

            }

            function on_assistant_action(received_dic){

                var is_done = false;

                action = unpack_actions(received_dic);
                console.log("assistant_action", action);

                pointing_task_manager.assistant_action = action;
                console.log("task  dynamics", pointing_task_manager.position, pointing_task_manager.user_action, pointing_task_manager.assistant_action);
                pointing_task_manager.position += pointing_task_manager.user_action*pointing_task_manager.assistant_action;
                return_dic = {"state": pointing_task_manager.state, "reward": -.5, "is_done": false};
                send_msg(return_dic);

            }



            // ========================== Start ========================

            var minus = document.querySelector('.minus'),
            plus = document.querySelector('.plus'),
            users = document.querySelector('.users'),
            // targets = document.querySelector('.targets'),
            // position = document.querySelector('.position'),
            goal = document.querySelector('.goal'),
            parameters = document.querySelector('.param'),
            websocket = new WebSocket("ws://127.0.0.1:4000/");


            var users_actions = {};
            var iter = 0;


            websocket.onmessage = function (event) {
                console.log(event)
                let dic_received = JSON.parse(event.data);
                let data_type = dic_received.type
                delete dic_received.type
                switch (data_type) {
                    case 'init':
                        on_init(dic_received);
                        break;
                    case 'reset':
                        on_reset(dic_received);
                        break;
                    case 'user_action':
                        on_user_action(dic_received);
                        break;
                    case 'assistant_action':
                        on_assistant_action(dic_received);
                        break;
                    case 'state':
                        update_state(dic_received);
                        break;
                    case 'done':
                        break;

                    default:
                        console.error(
                            "unsupported event", data);
                }
            };
